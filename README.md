# [ICCV 2025] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion
<p align="center"> 
  <img src="./assets/TEASER7.png">

</p>


<div align="center">
  <a href=https://zebinhe.github.io/MaterialMVP/ target="_blank"><img src= https://img.shields.io/badge/Project%20page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://arxiv.org/abs/2503.10289 target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv height=22px></a>
</div>

<!-- **[MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](#)**

Zebin He<sup>1,2,*</sup>, 
Mingxin Yang<sup>2;</sup>, 
Shuhui Yang<sup>2;</sup>, 
Yixuan Tang<sup>2;</sup>, 
[Tao Wang](https://taowangzj.github.io/)<sup>3;</sup>, 
[Kaihao Zhang](https://zhangkaihao.github.io/)<sup>4;</sup>, 
[Guanying Chen](https://guanyingc.github.io/)<sup>1;</sup>, 
Yuhong Liu<sup>2</sup>, 
Jie Jiang<sup>2</sup>, 
Chunchao Guo<sup>2&dagger;</sup>, 
[Wenhan Luo](https://whluo.github.io/)<sup>5&#9993;</sup>

1 Sun Yat-sen University (Shenzhen) \
2 Tencent Hunyuan \
3 Nanjing University \
4 Harbin Institute of Technology (Shenzhen) \
5 The Hong Kong University of Science and Technology

\* Intern at Hunyuan3D, Tencent \
&dagger; Project Leader \
&#9993; Corresponding Author -->


## News

- Jul 2, 2025: Code release.
- Jun 26, 2025: MaterialMVP is accepted by ICCV 2025.
- Mar 14, 2025: [arXiv](https://arxiv.org/abs/2503.10289) preprint is now available.


## Quick Start
### 1. Installation
```bash
git clone -b main --single-branch --depth 1 https://github.com/ZebinHe/MaterialMVP.git
cd MaterialMVP

conda create -n materialmvp python=3.10 -y
conda activate materialmvp
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

cd custom_rasterizer
pip install -e .
cd ..
cd DifferentiableRenderer
bash compile_mesh_painter.sh
cd ..

wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ckpt
```
### 2. Model Preparation.
Two model weights are available: the professional version (i.e. Hunyuan3D-Paint) for the best performance, and the paper version for academic comparisons.
| Model | Using DINO | Resolution | Training Data | Huggingface |
|:-----:|:--------:|:----------:|:-------------:|:-----------:| 
| Professional Version<br>(Hunyuan3D-Paint-v2-1) | True | 768 and 512 | Private Dataset | [Download](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-paint-v2-1) |
| Paper Version | False | 512 | Objaverse<br>Objaverse-XL | coming soon |

### 3. Quick Inference
```python
from textureGenPipeline import MaterialMVPPipeline, MaterialMVPConfig

pipe = MaterialMVPPipeline(MaterialMVPConfig(max_num_view=6, resolution=512))
textured_mesh = pipe(mesh_path="test_examples/mesh.glb", image_path="test_examples/image.png")
```
Try `demo.py` instead if get a ModuleNotFoundError from torchvision. The result will be saved as `textured_mesh.glb`.

## Training

### Data Prepare
We provide a piece of data in `train_examples` for the overfitting training test. The data structure should be organized as follows:

```
train_examples/
├── examples.json
└── 001/
    ├── render_tex/                 # Rendered generated PBR images
    │   ├── 000.png                 # Rendered views (RGB images)
    │   ├── 000_albedo.png          # Albedo maps for each view
    │   ├── 000_mr.png              # Metallic-Roughness maps for each view, R and G channels
    │   ├── 000_normal.png          # Normal maps
    │   ├── 000_normal.png          # Normal maps
    │   ├── 000_pos.png             # Position maps
    │   ├── 000_pos.png             # Position maps
    │   ├── 001.png                 # Additional views...
    │   ├── 001_albedo.png
    │   ├── 001_mr.png
    │   ├── 001_normal.png
    │   ├── 001_pos.png
    │   └── ...                     # More views (002, 003, 004, 005, ...)
    └── render_cond/                # Rendered reference images (at least two light conditions should be rendered to facilitate consistency loss)
        ├── 000_light_AL.png        # Light condition 1 (Area Light)
        ├── 000_light_ENVMAP.png    # Light condition 2 (Environment map)
        ├── 000_light_PL.png        # Light condition 3 (Point lighting)
        ├── 001_light_AL.png        
        ├── 001_light_ENVMAP.png
        ├── 001_light_PL.png
        └── ...                      # More lighting conditions (002-005, ...)
```

Each training example contains:
- **render_tex/**: Multi-view renderings with PBR material properties
  - Main RGB images (`XXX.png`)
  - Albedo maps (`XXX_albedo.png`)
  - Metallic-Roughness maps (`XXX_mr.png`)
  - Normal maps (`XXX_normal.png/jpg`)
  - Position maps (`XXX_pos.png/jpg`)
  - Camera transforms (`transforms.json`)
- **render_cond/**: Lighting condition maps for each view
  - Ambient lighting (`XXX_light_AL.png`)
  - Environment map lighting (`XXX_light_ENVMAP.png`)
  - Point lighting (`XXX_light_PL.png`)

### Launch Training

```bash
python3 train.py --base 'cfgs/v1.yaml' --name overfit --logdir logs/
```

## BibTeX

If you found MaterialMVP helpful, please cite our paper:
```bibtex
@article{he2025materialmvp,
  title={MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion},
  author={He, Zebin and Yang, Mingxin and Yang, Shuhui and Tang, Yixuan and Wang, Tao and Zhang, Kaihao and Chen, Guanying and Liu, Yuhong and Jiang, Jie and Guo, Chunchao and Luo, Wenhan},
  journal={arXiv preprint arXiv:2503.10289},
  year={2025}
}
```

## Links
- [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [RomanTex](https://github.com/oakshy/RomanTex) (RomanTex is also accepted by ICCV 2025. Congrats!)
