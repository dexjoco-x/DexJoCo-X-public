"""Task-local collection/replay acceptance; does not change native physics.

Bucket and microwave end on the FIRST post-control-step native success. Other tasks retain
their existing ten-step acceptance tail. Old reports are never relabeled.
"""


def canonical_task(task):
    return {'bucket': 'pick_bucket', 'hammer': 'hammer_nail'}.get(task, task)


def required_success_steps(task):
    if canonical_task(task) == 'single_hand_hanoi':return 1
    return 1 if canonical_task(task) in ('pick_bucket', 'bimanual_microwave_cook') else 10


def success_policy(task):
    if canonical_task(task) == 'single_hand_hanoi':
        return dict(id='single_hand_hanoi_released_v1', required_success_steps=1,
                    terminate_on_first_success=True, native_stable_steps=20,
                    released_stable_steps=20, active_arms=['right'],
                    left_task_contact_invalidates_episode=True)
    if canonical_task(task) == 'bimanual_microwave_cook':
        return dict(id='microwave_first_native_success_v1',
                    required_success_steps=1, terminate_on_first_success=True,
                    observation='post_control_step_50ms',
                    completion='native food-inside AND door-closed AND start-button-contact',
                    additional_hold_s=0.0)
    if canonical_task(task) == 'pick_bucket':
        return dict(id='pick_bucket_first_native_success_v1',
                    required_success_steps=1, terminate_on_first_success=True,
                    observation='post_control_step_50ms',
                    food_condition='recorded food position inside bucket reference AABB',
                    height_condition='all bucket bottom reference points lifted from reset',
                    minimum_lift_m=0.15, additional_hold_s=0.0)
    return dict(id='native_success_stable10_v1', required_success_steps=10,
                terminate_on_first_success=False)


def policy_matches(task, record):
    if canonical_task(task) == 'single_hand_hanoi':
        return record.get('success_policy') == success_policy(task)
    # Historical non-bucket releases have no policy field; their gate is unchanged.
    return (canonical_task(task) != 'pick_bucket'
            or record.get('success_policy') == success_policy(task))


def validate_success_sequence(task, values):
    """Validate actual replay labels, not a summary's claimed success count."""
    values = [bool(value) for value in values]
    required = required_success_steps(task)
    if len(values) < required or not all(values[-required:]):
        raise ValueError(f'Missing terminal success: requires {required} step(s)')
    first = values.index(True)
    if canonical_task(task) in ('pick_bucket', 'single_hand_hanoi') and first != len(values) - 1:
        raise ValueError('Trajectory continues after terminal success')
    return first
