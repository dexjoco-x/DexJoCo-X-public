# LinkerHand model provenance

The source URDFs in this directory are derived from the official LinkerHand **L25** model, not the official L20 model:

- upstream: https://github.com/linker-bot/linkerhand-urdf
- audited upstream commit: `075cc7d42cc1e756bdcbece0fc069a0779fc5237`
- upstream files: `L25/right/linkerhand_l25_right.urdf` and `L25/left/linkerhand_l25_left.urdf`
- license: Apache-2.0; the upstream license is copied as `SOURCE_LICENSE`

Relative to the audited upstream L25 URDF, each local `source.urdf` changes the robot name and reverses four joint-axis signs. Generated MuJoCo assets add mounting, actuator, and simulator-specific changes. The historical internal identifier `linkerhand_l20` is retained for checkpoint and manifest compatibility; public prose should identify this embodiment as `LinkerHand` until the hardware/model designation is resolved.
