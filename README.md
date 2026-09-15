# MoPo-I2V: Motion-Posterior Image-to-Video Generation with Geometric Uncertainty for 4D Medical Imaging

Official code for the paper **MoPo-I2V: Motion-Posterior Image-to-Video Generation with Geometric Uncertainty for 4D Medical Imaging**.

This repository provides MoPo-I2V, a motion-first image-to-video framework for 4D medical imaging (cardiac cine MRI, 4D cardiac CTA and 4D lung CT). Motion Generation samples the motion posterior with a non-autonomous flow-map diffusion under a composition-consistency constraint; Motion-Posterior Metamorphosis transports the first volume along the posterior mean and completes appearance with a pixel-space generative bridge guided by the uncertainty pushed forward from the posterior.

## Repository Structure

- `train_motion.py`: training of the motion-posterior generator
- `train_metamorphosis.py`: training of the pixel-space generative bridge
- `inference.py`: image-to-video generation from a single volume
- `evaluate.py`: metric evaluation (PSNR, LPIPS, FVD, motion accuracy)
- `models/`, `losses/`: model and loss components
- `data/`: dataset and experiment outputs

## Usage

### 1. Environment Setup

Recommended environment:

- Python 3.9
- PyTorch
- h5py
- SimpleITK
- NiBabel
- SciPy
- tqdm

### 2. Datasets

Public datasets used in this project:

- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) (cardiac cine MRI)
- [4D-Lung](https://www.cancerimagingarchive.net/collection/4d-lung/) (4D lung CT)

The remaining dataset (4D cardiac CTA) is private and is not included in this repository.

### 3. Training

Motion-posterior generator:

```bash
python train_motion.py
```

Generative bridge:

```bash
python train_metamorphosis.py
```

### 4. Inference

```bash
python inference.py
```

### 5. Evaluation

```bash
python evaluate.py
```

## Citation

Citation information can be added here after the paper is publicly available.

## Contact

For questions or collaboration, please contact: `zhangc31@mails.neu.edu.cn`

## Acknowledgements

We thank the authors and contributors of the following open-source projects:

- [UVI-Net](https://github.com/jungeun122333/UVI-Net)
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
