# LEAP Hand Assets

This directory vendors LEAP Hand URDF assets for DexJoCo hand-morphology
experiments.

Source:
- `UniDex/HandAdapter/urdf/base/Leap/right`
- `UniDex/HandAdapter/urdf/base/Leap/left`

The assets are intentionally kept in the LEAP model's native convention. Do not
rename joints to Allegro-style names or force the model into DexJoCo's current
23-dimensional Allegro action layout at the asset layer.

Native controllable joints:
- Right hand: `0` through `15`
- Left hand: `0` through `15`

The URDFs also contain fixed joints for the palm/wrist and fingertip marker
links. Environment code should explicitly choose controllable joints by name
rather than relying on URDF traversal order.

The original per-side `config.json` files are not copied here because they
contain Onshape API credentials and are not needed for local simulation.
