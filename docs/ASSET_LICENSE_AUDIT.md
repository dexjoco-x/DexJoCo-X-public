> Public source snapshot: 801 runtime files, no glove mappings or weights. The project owner requested direct publication without new upstream contact. Historical verification receipts also mention non-distributed files, separated in the receipt JSONs. Outstanding third-party terms remain unconfirmed.

# Asset license audit (2026-10-01)

This is a provenance audit for release preparation, not legal advice. “Verified” means a license file is present locally or an exact/near-exact upstream derivation has been identified. It does not replace approval by the project rights holders.

## Robot and hand assets

| Asset | Local path | Evidence | Status |
|---|---|---|---|
| Franka Panda | `simulator/dexjoco/sim/envs/xmls/franka_emika_panda` | Local Apache-2.0 `LICENSE` | Verified |
| Allegro | `simulator/dexjoco/sim/envs/xmls/wonik_allegro` | Local BSD-2-Clause-style `LICENSE` | Verified |
| XHand | `simulator/dexjoco/sim/envs/xmls/xhand` | 62 files (60 runtime meshes and two source URDFs) are byte-identical to official DexLatent commit `502750df34b9fa6a1b56a66481586887ec722f9c`. Its root MIT license matches local `SOURCE_LICENSE`; notices, pinned source URLs and receipts are retained in `licenses/third_party/dexlatent`. Generated MJCF changes remain separately documented. | Official source and declared MIT terms documented for the exact subset |
| Sharpa Wave | `simulator/teleoperation/sharpa_wave/sharpa-urdf-usd-xml` | Official Apache-2.0 `LICENSE.txt` and `NOTICE.txt` copied from [Sharpa upstream](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) | Verified |
| LinkerHand | `simulator/dexjoco/sim/envs/xmls/linkerhand_l20` | Local URDF is a modified near-exact derivative of official **L25**, licensed Apache-2.0 at [LinkerHand upstream](https://github.com/linker-bot/linkerhand-urdf); license copied locally | Verified license; identity mismatch unresolved |
| LEAP | `simulator/dexjoco/sim/envs/xmls/leap_hand` | URDFs are byte-identical to UniDex `HandAdapter/urdf/base/Leap`. At UniDex commit `97d869e...`, its README shows an MIT badge linking to `./LICENSE`, but that file is absent and no per-hand license/source notice is included. Sixteen local meshes are byte-identical to official `LEAP_Hand_Sim` commit `150bc3d...` and are covered by its MIT license, now retained locally. The remaining files still lack a complete source chain. | Partially verified; license review incomplete |
| Inspire | `simulator/teleoperation/inspire` and `benchmark6_assets/inspire` | Eight right-hand collision meshes are byte-identical to Dex-URDF commit `f5e7132...`, which identifies Inspire models as CC BY-NC-SA 4.0. The full license, upstream notice, attribution and exact file receipts are now retained in `licenses/third_party/inspire`. Another 68 converted/material-split/mirrored meshes are now verified against nine exact upstream GLBs and the collision originals; only the remaining URDF chain needs separate verification. | 76 runtime mesh files documented for non-commercial redistribution; remaining URDF chain unresolved |
| Wuji | `simulator/teleoperation/wuji` | UniDex publicly distributes a Wuji model but supplies no per-hand license/source notice and its linked root license file is absent. Current official [wuji-description](https://github.com/wuji-technology/wuji-description) is MIT, but local D5/UniDex-derived files are modified and have not been matched to that official revision. | License scope unresolved pending exact-source match |

## Task and scene assets referenced by the 42 benchmark models

| Root | Typical content | Local license evidence | Status |
|---|---|---|---|
| `camera`, `ipad`, `photograph`, `boxed_food_0`, `hot_dog_2` | task objects, textures, logos | None found | License scope unresolved |
| `table_arena` | table textures | 43/50 files are byte-identical to RoboSuite 1.4.1 and covered by its MIT license; seven added desktop textures remain unresolved | Partially verified |
| `rethink_mount` | robot pedestal | STL is byte-identical to RoboSuite 1.4.1 under MIT; OBJ/DAE appear converted from that source but conversion receipt is absent | Conditional |
| `bucket` | articulated bucket | Directory fingerprint matches GAPartNet/PartNet-Mobility style (`mobility.urdf`, `mobility_v2.json`, `semantics.txt`, UUID metadata). GAPartNet states CC BY-NC 4.0, but the exact object ID/source receipt has not yet been recovered. | Likely non-commercial; exact source pending |
| `robocasa_microwave` | microwave meshes/textures | RoboCasa code is MIT at [official upstream](https://github.com/robocasa/robocasa/blob/main/LICENSE), but none of the five local files is byte-identical to official repository commit `456174f...`; the exact asset source and terms still need verification | License scope unresolved |
| `benchmark6_assets/inspire` | converted Inspire meshes | Exact upstream GLB and mirror/triangle receipts | Documented CC BY-NC-SA 4.0 for the recorded subset |

## Findings that require a release decision

1. The internal identifier `linkerhand_l20` does not describe the source geometry: both local URDFs derive from official L25 files, with four joint-axis reversals. Keep the internal ID for checkpoint compatibility, but use “LinkerHand” in public prose until the embodiment designation is confirmed.
2. Non-commercial terms do not prohibit a non-commercial research release when attribution, modification notices and ShareAlike requirements are met. The verified Inspire subset follows CC BY-NC-SA 4.0. Missing source chains in the remaining Inspire, LEAP and Wuji files are a separate issue; project-level licensing cannot replace those grants.
3. Task-object assets still need exact source/license receipts. Recover those receipts, obtain permission, or exclude uncleared assets from the public release. A download interface must point to an authorized upstream distribution, not to a rehosted copy of an uncleared file.
4. A top-level project code license and a separate data license are still required after the third-party inventory is resolved.

## UniDex repository check

The official [UniDex repository](https://github.com/unidex-ai/UniDex) was checked at commit `97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`. It publicly includes `Ability`, `Allegro`, `Inspire`, `Leap`, `Oymotion`, `Shadow`, `Wuji`, and `Xhand` under `HandAdapter/urdf/base`. Its README displays an MIT badge linked to `./LICENSE`, but that target is absent from the commit, and no `LICENSE`, `COPYING`, or `NOTICE` file exists anywhere in the tree. Apart from BSD declarations inside OYMotion `package.xml` files, no per-hand license or provenance notices were found. Public availability is useful provenance evidence, but does not by itself provide a complete redistribution grant for the bundled hand assets.

The machine-readable file counts, byte totals, tree hashes, and source-match evidence are recorded in `ASSET_INVENTORY.json`.

## File-level release preparation

225 uncleared runtime task/scene assets are byte-identical to official DexJoCo commit `8d23b0fab23b17a58c4b55f3942e17013aaf8267`; another 57 still-uncleared files are byte-identical to the audited UniDex snapshot (nine additional matches have now been cleared through Inspire derivation evidence). Exact file receipts are in `ASSET_SOURCE_MATCHES.json`. This confirms their upstream location, not their asset redistribution terms. DexJoCo's README explicitly assigns MIT to DexJoCo-owned code and preserves separate third-party asset terms. Its full copyright/license notice and the scope statement are now retained in `licenses/third_party/dexjoco/`.

`licenses/third_party/*/VERIFIED_FILES.json` records exact upstream Git blob IDs, SHA-256 hashes, pinned source URLs and modification statements for verified subsets. These receipts do not assign a license to other files in the same directory.

Run `python tools/asset_release_report.py` to resolve the mesh/texture/include references of all 42 frozen evaluation models. `docs/ASSET_RELEASE_REPORT.json` lists each referenced file, every hand–task pair using it and its current license status. Use `--require-ready` to make uncleared runtime assets a failing check. This report does not clear all repository files or the RGB dataset: rendered images may incorporate separately licensed models and textures, so data licensing needs its own review.

The 68 derived runtime Inspire meshes now have complete conversion evidence against nine byte-identical pinned upstream GLBs and the eight original collision meshes. They are documented under CC BY-NC-SA 4.0 with modification statements. The dataset/evaluation snapshot includes frozen runtime model assets with their provenance records, excludes all policy/codec weights and model training implementation, and remains private. Technical validation is recorded in `DATASET_EVAL_RELEASE_VALIDATION.json`.

The sole previously untraced runtime texture, `photograph/logo-DIG-inspire.png`, was generated within this project. Its historical Pillow recipe reproduces the entire 4000×4000 RGB pixel buffer exactly. `PROJECT_ASSET_PROVENANCE.json` retains the original file and pixel hashes, font notice, and verifier hash. It is now covered by the first-party MIT scope, separately from third-party models. The remaining 282 third-party files all have pinned upstream content matches, but their asset license scope remains unresolved.
