# DreamWAM

Official implementation of **DreamWAM: Beyond RGB Future Prediction for World
Action Models**.

[Project Page](https://hustvl.github.io/DreamWAM/) | [Paper](https://github.com/hustvl/DreamWAM) | [Models](https://huggingface.co/hustvl/DreamWAM)

## Installation

Python 3.10 and CUDA 12.x are recommended.

```bash
conda create -n dreamwam python=3.10 -y
conda activate dreamwam
conda install -c conda-forge imagemagick -y
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -e third_party/Depth-Anything-3
pip install -e third_party/LIBERO
```

## External Components

Prepare the following upstream repositories and pretrained components under the
repository-local targets shown below.

| Component | Upstream source or weights | Local target |
| --- | --- | --- |
| DreamWAM checkpoints | [here](https://huggingface.co/hustvl/DreamWAM) | `checkpoints/` |
| Wan2.2 TI2V-5B | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `pretrained/Wan2.2-TI2V-5B/` |
| ActionDiT initialization | Derived from the Wan2.2 VideoDiT with `scripts/prepare_action_dit.py` | `pretrained/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` |
| RAFT | [source](https://github.com/princeton-vl/RAFT) and [official model download script](https://github.com/princeton-vl/RAFT/blob/master/download_models.sh) | `third_party/RAFT/`, `pretrained/raft-things.pth` |
| DINOv2 | [source](https://github.com/facebookresearch/dinov2) and [official ViT-B/14 register weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) | `third_party/dinov2/`, `pretrained/dinov2_vitb14_reg4_pretrain.pth` |
| Depth Anything 3 | [source](https://github.com/ByteDance-Seed/Depth-Anything-3) and [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | `third_party/Depth-Anything-3/`, `pretrained/da3-base/` |
| LIBERO | [benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) | `third_party/LIBERO/` |
| LIBERO-Plus | [benchmark](https://github.com/sylvestf/LIBERO-plus) and [assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus) | `third_party/LIBERO-Plus/` with assets under `libero/libero/assets/` |

Prepare the FastWAM-style ActionDiT backbone after placing the Wan2.2 weights:

```bash
python scripts/prepare_action_dit.py --config configs/dreamwam_joint.yaml
```

## Data Preparation

DreamWAM uses the four-suite, LeRobot v2.1 LIBERO release prepared by FastWAM:
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam).
Download and extract its four suite archives under `data/libero` so the suite
directories are available directly below that path. The dataset was prepared
with MuJoCo 3.3.2; use the same version for benchmark consistency.

The preprocessing implementation in `scripts/precompute_cache.py` builds Wan
RGB and optical-flow latents together with compressed DINO and Depth targets.
The resulting training cache is written to `cache/libero_2cam224` and is shared
by both released settings.

```bash
python scripts/precompute_cache.py --config configs/dreamwam_joint.yaml
```

## Training

```bash
accelerate launch --num_processes 8 scripts/train.py --config configs/dreamwam_uncond.yaml
accelerate launch --num_processes 8 scripts/train.py --config configs/dreamwam_joint.yaml
```

Training results are written to `outputs/uncond/final.pt` and
`outputs/joint/final.pt`, respectively.

## Evaluation

```bash
python scripts/eval_libero.py --config configs/dreamwam_joint.yaml --suite libero_spatial
python scripts/eval_libero_plus.py --config configs/dreamwam_joint.yaml
```

The LIBERO-Plus entry point follows the four-suite, 10,030-task protocol with one
trial per task. It reports per-suite, weighted, and seven-dimension perturbation
averages as JSON under `outputs/`.

## Acknowledgements

DreamWAM is built on the [FastWAM](https://github.com/yuantianyuan01/FastWAM.git)
codebase and its coupled VideoDiT-ActionDiT formulation. We thank the FastWAM
authors for releasing the base model, training pipeline, and processed LIBERO
data. We also acknowledge the upstream Wan2.2, RAFT, DINOv2, Depth Anything 3,
LIBERO, and LIBERO-Plus projects used by this release.
