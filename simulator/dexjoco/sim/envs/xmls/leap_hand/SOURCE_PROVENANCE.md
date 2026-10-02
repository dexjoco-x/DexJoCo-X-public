# LEAP Hand source provenance

The local right- and left-hand bundles were copied from
`UniDex/HandAdapter/urdf/base/Leap`. The audited UniDex repository does not
provide a root license, so that intermediate redistribution chain remains
unresolved.

An exact SHA-256 comparison against the official `LEAP_Hand_Sim` repository at
commit `150bc3d4b61fd6619193ba5a8ef209f3609ced89` found 16 byte-identical mesh
files: `thumb_dip.stl`, `dip.stl`, `palm_lower.stl`, `thumb_pip.stl`,
`fingertip_old.stl`, `pip.stl`, `thumb_fingertip_old.stl`, and `mcp_joint.stl`
for both local sides. The official source distributes these files under MIT;
its license is retained as `SOURCE_LICENSE_LEAP_HAND_SIM`.

The local URDFs and remaining meshes are not byte-identical to that official
revision. The included upstream license therefore documents only the verified
source subset and does not by itself establish the redistribution terms for
the complete local bundle.
