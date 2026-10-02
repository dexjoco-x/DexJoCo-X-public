"""Shared lift/hold/unlock gate; screen orientation is diagnostic only.

Development thresholds, not hardware calibration. The state machine is sampled
at each 2 ms physics step; reading metrics never advances it.
"""
from dataclasses import asdict, dataclass
import mujoco
import numpy as np

REVISION = 'left_lift_hold_then_unlock_no_orientation_v2'


@dataclass(frozen=True)
class Rules:
    revision: str = REVISION
    status: str = 'development_not_frozen'
    lift_m: float = .05
    support_clearance_m: float = .005
    target_screen_normal: tuple = (-1., 0., 0.)
    orientation_is_success_criterion: bool = False
    preparation_linear_speed: float = .08
    preparation_angular_speed: float = .5
    preparation_hold_s: float = .5
    contact_force_min_n: float = .02
    opposed_normal_dot_max: float = -.2
    contact_separation_min_m: float = .02
    hold_lift_m: float = .04


RULES = Rules()


class SequenceGate:
    def __init__(self, rules=RULES):
        self.rules = rules
        self.phase = 'acquire'
        self.failure = ''
        self.ready_since = None
        self.last_time = None
        self.history = []
        self.digits = 0

    @property
    def ready(self):
        return self.phase in ('ready', 'input', 'unlocked')

    def fail(self, reason, time):
        if not self.failure:
            self.failure = reason
            self.phase = 'failed'
            self.history.append(dict(phase='failed', time=float(time), reason=reason))

    def observe(self, m, time):
        time = float(time)
        if self.last_time is not None and time < self.last_time:
            raise ValueError('Nonmonotonic task clock')
        if time == self.last_time or self.failure: return
        self.last_time = time
        r = self.rules
        held = m['grasp_opposed'] and m['support_clearance_m'] >= r.support_clearance_m
        if self.ready:
            if not held or m['lift_m'] < r.hold_lift_m:
                self.fail('ipad_left_hold_lost', time)
            return
        if m['right_contact']:
            self.fail('ipad_right_contact_before_ready', time)
            return
        prepared = (held and m['lift_m'] >= r.lift_m and
            m['linear_speed'] <= r.preparation_linear_speed and
            m['angular_speed'] <= r.preparation_angular_speed)
        if not prepared:
            self.ready_since = None
            return
        if self.ready_since is None: self.ready_since = time
        if time - self.ready_since >= r.preparation_hold_s - 1e-10:
            self.phase = 'ready'
            self.history.append(dict(phase='ready', time=time))

    def mark_digit(self, count, time):
        if not self.ready or count != self.digits + 1 or count > 3:
            self.fail('ipad_invalid_phase_or_digit_transition', time)
            return
        self.digits = count
        self.phase = 'unlocked' if count == 3 else 'input'
        self.history.append(dict(phase=self.phase, digit_index=count, time=float(time)))


def opposed_contacts(points, rules=RULES):
    """Two distinct native finger/palm branches, not two points on one geom."""
    return any(a['branch'] != b['branch'] and
        np.linalg.norm(a['position']-b['position']) >= rules.contact_separation_min_m and
        np.dot(a['normal'], b['normal']) <= rules.opposed_normal_dot_max
        for i, a in enumerate(points) for b in points[i+1:])


class Measurements:
    def __init__(self, env):
        self.env = env
        m, d = env._model, env._data
        self.ipad = m.body('ipad_body').id
        self.screen = m.geom('ipad_screen').id
        self.initial_screen_z = float(d.geom_xpos[self.screen, 2])
        self.branches = {}
        self.groups = []
        for g in range(m.ngeom):
            body = int(m.geom_bodyid[g])
            ancestors = []
            while body:
                ancestors.append(body)
                body = int(m.body_parentid[body])
            names = [m.body(b).name for b in ancestors]
            if self.ipad in ancestors: group = 'ipad'
            elif any(n.startswith('benchhand_left_') for n in names): group = 'left'
            elif any(n.startswith('benchhand_right_') for n in names): group = 'right'
            else: group = 'environment'
            self.groups.append(group)
            if group == 'left' and 'adapter' not in m.geom(g).name:
                joints = [b for b in ancestors if m.body_jntnum[b] and m.body(b).name.startswith('benchhand_left_')]
                # The highest native joint-bearing ancestor identifies a finger;
                # the palm is another branch, while the wrist adapter is excluded.
                self.branches[g] = joints[-1] if joints else -1
        self.solid_ipad = [g for g, s in enumerate(self.groups) if s == 'ipad' and
            (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.supports = [g for g, s in enumerate(self.groups) if s == 'environment' and
            (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.maximum_lift = 0.
        self.maximum_left_contacts = 0
        self.grasp_samples = 0
        self.samples = 0
        self.last = {}

    def sample(self):
        env = self.env
        m, d = env._model, env._data
        left = []
        right = False
        support_contact = False
        force = np.zeros(6)
        for i, c in enumerate(d.contact):
            g1, g2 = int(c.geom1), int(c.geom2)
            if min(g1, g2) < 0: raise ValueError('Non-geom iPad contact unsupported')
            if 'ipad' not in (self.groups[g1], self.groups[g2]) or c.dist > 0: continue
            other = g2 if self.groups[g1] == 'ipad' else g1
            group = self.groups[other]
            if group == 'right': right = True
            if group == 'environment': support_contact = True
            if other not in self.branches or c.efc_address < 0: continue
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] < RULES.contact_force_min_n: continue
            normal = np.asarray(c.frame).reshape(3, 3)[0].copy()
            if self.groups[g2] == 'ipad': normal *= -1
            left.append(dict(branch=self.branches[other], position=np.asarray(c.pos).copy(), normal=normal))
        lift = float(d.geom_xpos[self.screen, 2] - self.initial_screen_z)
        # A touching support is already disqualifying. Exact separation checks
        # are needed only near/above the lift threshold, keeping reset cheap.
        clearance = 0.
        if not support_contact and lift >= RULES.hold_lift_m:
            clearance = min((float(mujoco.mj_geomDistance(m, d, a, b, .02, None))
                for a in self.solid_ipad for b in self.supports), default=.02)
        normal = d.geom_xmat[self.screen].reshape(3, 3)[:, 2]
        error = float(np.degrees(np.arccos(np.clip(np.dot(normal, RULES.target_screen_normal), -1., 1.))))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.ipad, velocity, 0)
        result = dict(lift_m=lift, support_clearance_m=clearance, right_contact=right,
            grasp_opposed=bool(opposed_contacts(left)), left_contact_points=len(left),
            left_contact_branches=len({p['branch'] for p in left}), orientation_error_deg=error,
            linear_speed=float(np.linalg.norm(velocity[3:])), angular_speed=float(np.linalg.norm(velocity[:3])))
        if not all(np.isfinite(v) for v in result.values()): raise RuntimeError('Nonfinite iPad task measurement')
        self.maximum_lift = max(self.maximum_lift, lift)
        self.maximum_left_contacts = max(self.maximum_left_contacts, len(left))
        self.grasp_samples += int(result['grasp_opposed'])
        self.samples += 1
        self.last = result
        return result


def contract():
    return asdict(RULES) | dict(sampling='every physics substep',
        grasp_evidence='force-bearing opposed contacts on distinct left native branches',
        early_right_contact='terminal_failure', lost_hold_after_ready='terminal_failure',
        screen_orientation='diagnostic_only_never_success_or_failure')
