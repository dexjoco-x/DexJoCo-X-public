# DexJoCo-X

**English** | [中文](README.zh.md)

A multi-hand manipulation dataset and a unified simulation benchmark.

[Website](https://jxw-wei.github.io/DexJoCo-X-website/) · [Paper](https://arxiv.org/abs/2610.03278) · [Dataset](https://huggingface.co/datasets/darenrenjian/DexJoCo-X)

## News

- 🔧 **2026-10-28** — **Training v1** is the current dataset for training and replay: the same 2,100 trajectories in float64 and matching LeRobot/Being-H formats. [Download and usage](dataset/README.md).

- 🎉 **2026-10-12** — The benchmark code and simulation environments are publicly available on [GitHub](https://github.com/dexjoco-x/DexJoCo-X-public).
- 🎉 **2026-10-05** — The dataset of 2,100 demonstrations is released on [Hugging Face](https://huggingface.co/datasets/darenrenjian/DexJoCo-X).
- 🎉 **2026-10-02** — The paper's first version is submitted to [arXiv](https://arxiv.org/abs/2610.03278).

![Overview](https://jxw-wei.github.io/DexJoCo-X-website/assets/overview.webp)

**7 hands · 6 tasks · 2,100 demonstrations**

DexJoCo-X provides demonstrations, MuJoCo environments, and a common policy evaluation interface. Train your own policies and evaluate them under consistent task settings and success criteria. Required simulation models and assets are included.

The seven hands are Allegro, Inspire, LEAP, LinkerHand, Sharpa Wave, Wuji, and XHand, with 50 training demonstrations per hand–task pair.

| Type | Task | Configuration |
|---|---|---|
| Single-arm | Hammer a nail | `hammer_nail` |
| Single-arm | Lift a bucket | `pick_bucket` |
| Single-arm | Tower of Hanoi | `single_hand_hanoi` |
| Bimanual | Operate a microwave | `bimanual_microwave_cook` |
| Bimanual | Take a photograph | `bimanual_photograph` |
| Bimanual | Unlock a tablet | `bimanual_unlock_ipad` |

## Get started

Use the reference environment on Linux x86_64 with AVX2/FMA:

```bash
git clone https://github.com/dexjoco-x/DexJoCo-X-public.git
cd DexJoCo-X-public
conda create -n dexjoco-x --file requirements/native-replay-linux64.conda.txt
conda env config vars set -n dexjoco-x OPENBLAS_CORETYPE=Haswell \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
conda activate dexjoco-x
python -m pip install -r requirements/native-replay-pip.txt
python -m pip install --force-reinstall --no-deps numpy==1.26.4 PyOpenGL==3.1.10
pip install -c requirements/constraints-python311.txt \
  -e './simulator[benchmark,dataset,download]'
python tools/runtime_assets.py check
```

Follow the [quickstart](docs/QUICKSTART.md) to download Training v1, read a demonstration, and connect a policy. The downloader's `--dry-run` option reports the selected download size.

## Guides

| Goal | Guide |
|---|---|
| Install and read the first demonstration | [Quickstart](docs/QUICKSTART.md) |
| Download data, inspect fields, convert formats | [Dataset](dataset/README.md) |
| Create and use simulation environments | [Environments](simulator/README.md) |
| Connect a policy and evaluate | [Evaluation](evaluation/README.md) |
| Connect π₀.₅ / Ego-π | [Policy example](examples/pi05/README.md) |
| Expand scenes from existing demonstrations | [Scene expansion](collection/README.md) |
| Check data and environment compatibility | [Versions](docs/VERSIONING.md) |
| Reproduce saved demonstration actions | [Replay setup](docs/REPLAY_REPAIR.md) |

Training implementations, policy weights, and glove recording tools are managed separately.

## Contact

For questions, feedback, or help using DexJoCo-X, contact [Xiangwei Jiang](mailto:jxw@std.uestc.edu.cn) at **jxw@std.uestc.edu.cn**, or open a [GitHub issue](https://github.com/dexjoco-x/DexJoCo-X-public/issues).

## Citation and licenses

See [BibTeX](CITATION.bib). Original code uses [MIT](LICENSE); original dataset contributions use [CC BY-NC-SA 4.0](dataset/LICENSE_SCOPE.md). Third-party components retain their own terms; see [notices](THIRD_PARTY_NOTICES.md) and [license status](LICENSE_STATUS.md).
