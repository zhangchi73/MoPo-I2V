# MoPo-I2V: Motion-Posterior Image-to-Video Generation with Geometric Uncertainty for 4D Medical Imaging

Official code for the paper **MoPo-I2V: Motion-Posterior Image-to-Video Generation with Geometric Uncertainty for 4D Medical Imaging**.

This repository provides MoPo-I2V, a motion-first image-to-video framework for 4D medical imaging (cardiac cine MRI, 4D cardiac CTA and 4D lung CT). Motion Generation samples the motion posterior with a non-autonomous flow-map diffusion under a composition-consistency constraint; Motion-Posterior Metamorphosis transports the first volume along the posterior mean and completes appearance with a pixel-space generative bridge guided by the uncertainty pushed forward from the posterior.

## Repository Structure

Motion Generation (Stage 1)

- `motion/arch.py`: non-autonomous flow-map diffusion network (MAISI latent U-Net backbone + temporal attention + LoRA + phase / amplitude conditioning)
- `motion/cocycle.py`: composition-consistency (cocycle) loss on flow-map pairs
- `motion/spatial.py`: displacement-field convention, warp and composition operators
- `motion/data.py`: window dataset, conditioning and rectified-flow sampling with classifier-free guidance
- `train_motion.py`: Stage-1 training
- `sample_motion.py`: sample N flow maps per case from the motion posterior

Motion-Posterior Metamorphosis (Stage 2)

- `transport/bspline.py`: 3D cubic B-spline transport of the first volume along a flow map
- `build_posterior_cache.py`: transports along the posterior samples and the posterior mean, and the pushed-forward uncertainty sigma^2
- `metamorphosis/unet_g.py`, `train_g.py`, `export_mu.py`: deterministic head G (residual mean, 1/sigma^2-weighted)
- `metamorphosis/unet_f.py`, `train_f.py`: generative bridge F (flow-matching bridge with uncertainty-guided noise, consistency and adversarial terms)
- `sample_f.py`: image-to-video inference
- `evaluate.py`: PSNR / SSIM / LPIPS / motion-magnitude evaluation

## Usage

### 1. Environment Setup

- Python 3.10
- PyTorch >= 2.0, MONAI >= 1.4 (for the MAISI diffusion U-Net and the rectified-flow scheduler)
- NumPy, SciPy, h5py, scikit-image, lpips

```bash
pip install -r requirements.txt
```

The Stage-1 backbone is initialised from the NV-Generate-MR latent diffusion U-Net (`diff_unet_3d_rflow-mr.pt` and `config_network_rflow.json`, available from NVIDIA); the anchor-frame latents `z0` are produced by its VAE.

### 2. Datasets

Public datasets used in this project:

- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) (cardiac cine MRI)
- [4D-Lung](https://www.cancerimagingarchive.net/collection/4d-lung/) (4D lung CT)

The remaining dataset (4D cardiac CTA) is private and is not included in this repository.

Each case is stored as `<case_id>.h5` with a dataset `image` of shape `[T, Z, Y, X]` (float32, intensities in `[0, 1]`). Stage-1 training additionally needs a field cache (one `.npz` per case with the teacher displacement fields of each sliding window, the anchor-frame latent and the window conditions; see the docstring of `motion/data.py` for the exact keys).

### 3. Training

Stage 1, Motion Generation:

```bash
python train_motion.py --cache data/field_cache --out runs/motion \
    --base-ckpt NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt \
    --base-cfg  NV-Generate-MR/configs/config_network_rflow.json --lam-coc 1.7
python sample_motion.py --ckpt runs/motion/last.pt --cache data/field_cache --split train \
    --base-ckpt ... --base-cfg ... --out data/posterior --nseed 5 --w 4
python build_posterior_cache.py --posterior data/posterior --h5dir data/acdc_h5 --split train --out data/mpm_cache
```

Stage 2, Motion-Posterior Metamorphosis:

```bash
python train_g.py --cache data/mpm_cache --out runs/g
python export_mu.py --run runs/g --cache data/mpm_cache --split train
python export_mu.py --run runs/g --cache data/mpm_cache --split val
python train_f.py --cache data/mpm_cache --mu runs/g --out runs/f
```

### 4. Inference

```bash
python sample_motion.py --ckpt runs/motion/last.pt --cache data/field_cache --split test --base-ckpt ... --base-cfg ... --out data/posterior --nseed 5 --w 4
python build_posterior_cache.py --posterior data/posterior --h5dir data/acdc_h5 --split test --out data/mpm_cache --single-window
python export_mu.py --run runs/g --cache data/mpm_cache --split test
python sample_f.py --run runs/f --cache data/mpm_cache --mu runs/g --split test --name mpm --seeds 0,1,2,3,4
```

### 5. Evaluation

```bash
python evaluate.py --npz-dir cells/mpm_r0 --h5dir data/acdc_h5 --out results/mpm_r0.json
```

## Citation

Citation information can be added here after the paper is publicly available.

## Contact

For questions or collaboration, please contact: `zhangc31@mails.neu.edu.cn`

