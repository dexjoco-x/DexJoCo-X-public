# Official camera profile v5

Profile: `dexjoco_official_allegro60_hanoi_front_v1`. Allegro official single/dual camera extrinsics are transferred through the attachment_site TCP frame to all seven hands. FOV 60 degrees. Hanoi uses the official single-arm front camera at [1.3, 0, 1.6] m, FOV 45. RGB stays 256x256 at 20 Hz, pre-action, no manual image rotation or flip.

For collection/evaluation using an existing simulation package, put this release tools directory on sys.path and call `from official_camera import activate; activate()` BEFORE creating benchmark6 environments. Alternatively call `configure(env)` for every already constructed environment. This affects only the current process. XHand photo and iPad still require their previously verified left-hand model revisions.

Local benchmark6/cameras.py now defaults to this profile. Historical data/checkpoints require `BENCHMARK7_CAMERA_PROFILE=common_tcp_side85_v1` before importing the updated local package. Never evaluate an old-camera checkpoint with the new-camera profile without explicitly intending that distribution change.

Use this release's own prepared directory and statistics for new training. Do not reuse v4 image caches. This is alignment to official simulation camera conventions, not proof of real Franka hand-eye calibration.

## Exact demonstration replay

Recorded-state rendering preserves all saved joint/object states. iPad additionally executes the original actions to restore its button highlights and unlocked screen. The older iPad sources require the original multithread BLAS numerical behavior: four BLAS threads reproduced the original default-32-thread reference within the existing 1e-8 acceptance tolerance; forcing one thread can diverge at contact. Newly recollected Allegro iPad was verified with one thread. These are numerical replay settings, not changes to robot stiffness, friction, or task rules. Per-episode camera conversion checks record actual replay errors.

## Frame interval

Every new video contains exactly the existing manifest steps. The effective training interval and native tensors are unchanged. Some v4 files stored extra post-success frames beyond the effective interval; those unused tail frames are not rendered into new wrist videos. Unchanged third-person video files are retained byte-for-byte and are still bounded by the manifest effective frame count when training.
