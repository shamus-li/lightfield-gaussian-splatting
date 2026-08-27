# Sparse Light Field Sampling Improves Casual 3D and 4D Reconstruction

[Shamus Li](https://shamus.li)<sup>1</sup>,
[Ruiming Cao](https://rmcao.net/)<sup>2</sup>,
[Laura Waller](https://www.laurawaller.com/)<sup>3</sup>,
[Kristina Monakhova](https://kristinamonakhova.com/)<sup>1</sup>,
[Sara Fridovich-Keil](https://sarafridov.github.io/)<sup>4</sup>

<sup>1</sup>Cornell University · <sup>2</sup>Adobe · <sup>3</sup>UC Berkeley ·
<sup>4</sup>Georgia Tech

[Project page](https://shamus.li/lightfield-gaussian-splatting/)

![Static and dynamic multi-view reconstruction results](docs/static/1-teaser.png)

Many consumer smartphones, stereo cameras, and light field cameras record multiple synchronized viewpoints in a single exposure event. However, novel view synthesis pipelines commonly use only a monocular stream and rely on camera motion or learned priors to obtain angular coverage. In this paper, we ask: why do we use only one viewpoint? We analyze *sensor-limited multi-view*, where one sensor trades off spatial and angular resolution, and *exposure-limited multi-view*, where multiple sensors on one commodity device observe each event simultaneously. We introduce a new dataset incorporating three types of commodity multi-view cameras, and evaluate sparse-view 3DGS and 4DGS baselines measuring reconstruction quality as a function of number of exposures and angle between extreme views. Our results demonstrate that using multiple cameras—even with a low baseline—significantly improves reconstruction quality in single-shot, few-shot, and casual video settings. In addition, under a fixed sensor budget, angular sampling improves reconstruction when exposures are scarce despite lower spatial resolution. The gains are most pronounced for single-shot and dynamic scenes, where a stationary monocular camera lacks the angular diversity to recover scene geometry and motion.

## Installation

We used Linux with CUDA 12.8.

```bash
git clone --recursive https://github.com/shamus-li/lightfield-gaussian-splatting.git
cd lightfield-gaussian-splatting
uv sync
```

Static evaluation also needs the RAFT weights used for covisibility masks:

```bash
./download_models.sh covisible
```

The baselines and the video pipelines need their own environments:

```bash
./setup.sh video      # dynamic reconstruction and Difix3D+
./setup.sh fsgs       # FSGS baseline
./setup.sh sparsegs   # SparseGS baseline
```

## Data

We collect and release two real-world datasets comprising 16 scenes (8 static, 8 dynamic), along with rendered inputs for the synthetic experiments. Download the [public dataset folder](https://drive.google.com/drive/folders/1D68RDdv5NK0_aY1kH_39CO9IjxuJBWja?usp=sharing) and extract its contents into `data/`.

```text
data/
├── casual/<scene>/
│   ├── iphone-train/
│   ├── iphone-eval/
│   ├── stereo-train/
│   ├── stereo-eval/
│   └── static/
│       ├── wide.png
│       ├── tele.png
│       ├── ultrawide.png
│       ├── stereo_left.png
│       ├── stereo_right.png
│       └── lightfield/
├── dynamic/<scene>/
│   ├── iphone/
│   └── stereo/
├── multiplexed/
│   ├── measurement.png
│   └── calibration.npz
└── synthetic/
    ├── blender/<scene>.blend
    └── <scene>/
        ├── transforms_train.json
        ├── transforms_test.json
        └── <camera>/<exposures>views/train/
            ├── transforms_train.json
            └── *.png
```

## Running the code

Prepare and train a real static scene:

```bash
uv run python train.py prepare-static \
  --data data/casual/action-figure

uv run python train.py static \
  --data data/casual/action-figure \
  --result-dir outputs/static/action-figure \
  --camera-model stereo
```

Train on a casual video capture:

```bash
uv run python train.py casual \
  --data data/casual/action-figure \
  --result-dir outputs/casual/action-figure \
  --camera-model iphone
```

Train on a dynamic capture:

```bash
uv run python train.py dynamic \
  --data data/dynamic/scissors \
  --result-dir outputs/scissors \
  --camera-model stereo
```

This writes `cfg.yml`, `prepared/` (frames per camera and the 4DGaussians dataset), `model/`
(4DGaussians config, checkpoint and test renders) and `metrics.json`. Add `--method monocular` to
train on the wide (iPhone) or right (stereo) stream alone, `--initializer vggt` to keep the cameras
static instead of tracking them with MegaSaM, and `--eval` to render and score a finished run again.

Train on a synthetic scene:

```bash
uv run python train.py synthetic \
  --data data/synthetic/drums \
  --result-dir outputs/drums \
  --camera-model stereo \
  --num-exposures 3
```

Train on one multiplexed image:

```bash
uv run python train.py multiplexed \
  --data data/multiplexed \
  --result-dir outputs/multiplexed
```

This writes `cfg.yml`, `model.pt`, `metrics.json` (sensor fit) and `renders/` (lens views,
predicted sensor image and residual, orbit video, `point_cloud.ply`). To render a finished run
again:

```bash
uv run python train.py multiplexed-render \
  --result-dir outputs/multiplexed \
  --output-dir outputs/multiplexed/renders-again
```

Prepare and train for depth rendering:

```bash
uv run python train.py prepare-depth \
  --data data/synthetic/drums

uv run python train.py depth \
  --data data/synthetic/drums \
  --result-dir outputs/drums
```

For commands with reference-image evaluation, pass `--eval` to recompute
evaluation for an existing result directory.

## Citation

```bibtex
@inproceedings{li2026sparse,
  title     = {Sparse Light Field Sampling Improves Casual 3D and 4D Reconstruction},
  author    = {Li, Shamus and Cao, Ruiming and Waller, Laura and Monakhova, Kristina and Fridovich-Keil, Sara},
  booktitle = {European Conference on Computer Vision Workshops},
  year      = {2026}
}
```
