"""Headless checks for the standalone O6/L6/L30 assets (MuJoCo 3.4.0).

Run: python -m unittest discover -s tests -p test_linkerhand_assets.py -v
The individual-joint test isolates the linkage from contacts; the full-hand
stress test keeps the shipped collision geometry and contact settings enabled.
"""
import hashlib
import json
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

ASSETS = Path(__file__).resolve().parents[1] / 'simulator/dexjoco/sim/envs/xmls'
HANDS = ('linkerhand_o6', 'linkerhand_l6', 'linkerhand_l30')
SIDES = ('left', 'right')


def coupling_error(model, data):
    follower = data.qpos[model.jnt_qposadr[model.eq_obj1id]]
    driver = data.qpos[model.jnt_qposadr[model.eq_obj2id]]
    expected = np.array([
        np.polynomial.polynomial.polyval(q, coef[:5])
        for q, coef in zip(driver, model.eq_data)
    ])
    return float(np.max(np.abs(follower - expected)))


class StandaloneLinkerHandAssets(unittest.TestCase):
    def load(self, hand, side):
        return mujoco.MjModel.from_xml_path(str(ASSETS / hand / side / 'hand.xml'))

    def advance(self, model, data, control, seconds):
        peak = 0.
        steps = round(seconds / model.opt.timestep)
        start = data.time
        for i in range(steps):
            data.ctrl[:] = control(i / max(steps - 1, 1))
            mujoco.mj_step(model, data)
            # Check every step: a numerical reset must not erase a failure.
            self.assertAlmostEqual(data.time, start + (i + 1) * model.opt.timestep, places=7)
            self.assertTrue(np.isfinite(data.qpos).all())
            self.assertTrue(np.isfinite(data.qvel).all())
            self.assertFalse(any(w.number for w in data.warning), str(data.warning))
            peak = max(peak, coupling_error(model, data))
        return peak

    def test_provenance_hashes(self):
        for hand in HANDS:
            directory = ASSETS / hand
            manifest = json.loads((directory / 'SOURCE_PROVENANCE.json').read_text())
            for entry in manifest['files']:
                with self.subTest(hand=hand, path=entry['path']):
                    data = (directory / entry['path']).read_bytes()
                    self.assertEqual(len(data), entry['bytes'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])

    def test_joint_ranges_and_actuator_order(self):
        for hand in HANDS:
            meta = yaml.safe_load((ASSETS / hand / 'meta.yaml').read_text())
            for side in SIDES:
                with self.subTest(hand=hand, side=side):
                    model = self.load(hand, side)
                    urdf = ET.parse(ASSETS / hand / side / 'source.urdf').getroot()
                    joints = {j.attrib['name']: j for j in urdf.findall('joint')}
                    expected = [n.format(**meta['variants'][side]['vars']) for n in meta['actuated_joints']]
                    self.assertEqual([model.joint(int(j)).name for j in model.actuator_trnid[:, 0]], expected)
                    extension = urdf.find('mujoco')
                    if hand == 'linkerhand_l6':
                        xml = ET.parse(ASSETS / hand / side / 'hand.xml').getroot()
                        self.assertEqual(extension.find('option').attrib, xml.find('option').attrib)
                        self.assertEqual([j.attrib for j in extension.findall('equality/joint')],
                                         [j.attrib for j in xml.findall('equality/joint')])
                    for joint_id in range(model.njnt):
                        joint = model.joint(joint_id)
                        limit = joints[joint.name].find('limit')
                        np.testing.assert_allclose(joint.range, [float(limit.get('lower')), float(limit.get('upper'))], atol=1e-8)
                    for aid, jid in enumerate(model.actuator_trnid[:, 0]):
                        self.assertTrue(model.actuator_ctrllimited[aid])
                        np.testing.assert_allclose(model.actuator_ctrlrange[aid], model.jnt_range[jid], atol=1e-8)
                    for eq in range(model.neq):
                        follower = model.joint(model.eq_obj1id[eq])
                        driver = model.joint(model.eq_obj2id[eq])
                        samples = np.polynomial.polynomial.polyval(np.linspace(*driver.range, 101), model.eq_data[eq, :5])
                        self.assertGreaterEqual(samples.min(), follower.range[0] - 1e-8)
                        self.assertLessEqual(samples.max(), follower.range[1] + 1e-8)
                        mimic = joints[follower.name].find('mimic')
                        self.assertEqual(mimic.get('joint'), driver.name)
                        self.assertAlmostEqual(float(mimic.get('multiplier', '1')), model.eq_data[eq, 1])

    def test_individual_fingers_follow_and_release(self):
        for hand in HANDS:
            for side in SIDES:
                model = self.load(hand, side)
                model.geom_contype[:] = 0
                model.geom_conaffinity[:] = 0
                for eq in range(model.neq):
                    jid = int(model.eq_obj2id[eq])
                    aid = int(np.flatnonzero(model.actuator_trnid[:, 0] == jid)[0])
                    with self.subTest(hand=hand, side=side, joint=model.joint(jid).name):
                        data = mujoco.MjData(model)
                        target = np.zeros(model.nu)
                        target[aid] = min(.8, .8 * model.actuator_ctrlrange[aid, 1])
                        peak = self.advance(model, data, lambda t: target * t, 3.)
                        self.assertLess(peak, .02, 'linkage lags during a slow closing command')
                        self.advance(model, data, lambda t: target, 5.)
                        self.assertLess(coupling_error(model, data), .001)
                        self.assertLess(abs(data.qpos[model.jnt_qposadr[jid]] - target[aid]), .03)
                        peak = self.advance(model, data, lambda t: target * (1 - t), 3.)
                        self.assertLess(peak, .02)
                        self.advance(model, data, lambda t: np.zeros(model.nu), 5.)
                        self.assertLess(coupling_error(model, data), .001)
                        self.assertLess(abs(data.qpos[model.jnt_qposadr[jid]]), .03)

    def test_full_hand_commands_with_contacts(self):
        # A commanded posture may be obstructed by self-contact. Check numerical
        # stability and linkage error, not exact servo tracking through a contact.
        for hand in HANDS:
            for side in SIDES:
                model = self.load(hand, side)
                for mode in ('ramp', 'step', 'random'):
                    with self.subTest(hand=hand, side=side, mode=mode):
                        data = mujoco.MjData(model)
                        upper = model.actuator_ctrlrange[:, 1].copy()
                        if mode == 'ramp':
                            peak = self.advance(model, data, lambda t: upper * t, 3.)
                        elif mode == 'step':
                            peak = self.advance(model, data, lambda t: upper, 3.)
                        else:
                            rng = np.random.default_rng(4)
                            peak = 0.
                            for _ in range(10):
                                target = rng.uniform(*model.actuator_ctrlrange.T)
                                peak = max(peak, self.advance(model, data, lambda t: target, .3))
                        peak = max(peak, self.advance(model, data, lambda t: upper, 7.))
                        self.assertLess(peak, .1)
                        self.assertLess(coupling_error(model, data), .01)
                        self.advance(model, data, lambda t: upper * (1 - t), 3.)
                        self.advance(model, data, lambda t: np.zeros(model.nu), 5.)
                        self.assertLess(coupling_error(model, data), .01)


if __name__ == '__main__':
    unittest.main()
