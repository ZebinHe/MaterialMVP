<p align="center"> 
  <img src="./assets/teaser.mp4">

</p>

<div align="center">
  <a href=https://3d-models.hunyuan.tencent.com/ target="_blank"><img src= https://img.shields.io/badge/Homepage-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://arxiv.org/abs/2501.12202 target="_blank"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
</div>

## 🔥 News

- Mar 13, 2025: 

## **Abstract**

Physically-based rendering (PBR) has become a cornerstone in modern computer graphics, enabling realistic material representation and lighting interactions in 3D scenes. In this paper, we present MaterialMVP, a novel end-to-end model for generating PBR textures from 3D meshes and image prompts, addressing key challenges in multi-view material synthesis. Our approach leverages Reference Attention to extract and encode informative latent from the input reference images, enabling intuitive and controllable texture generation. We also introduce a Consistency-Regularized Training strategy to enforce stability across varying viewpoints and illumination conditions, ensuring illumination-invariant and geometrically consistent results. Additionally, we propose Dual-Channel Material Generation, which separately optimizes albedo and metallic-roughness (MR) textures while maintaining precise spatial alignment with the input images through Multi-Channel Aligned Attention. Learnable material embeddings are further integrated to capture the distinct properties of albedo and MR. Experimental results demonstrate that our model generates PBR textures with realistic behavior across diverse lighting scenarios, outperforming existing methods in both consistency and quality for scalable 3D asset creation.



<p align="center">
  <img src="assets/TEASER7.png">
</p>

## **MaterialMVP**

### Method

The proposed method introduces a framework for generating high-quality, view-consistent Physically Based Rendering (PBR) maps from image prompts using a Multiview Diffusion Model. At its core, the approach employs a Dual-Channel Material Generation framework, which extends the traditional diffusion model by incorporating an additional channel to simultaneously produce albedo and metallic-roughness (MR) maps. To ensure the generated textures retain fine details and remain faithful to the input reference image, we utilize Reference Attention, which extracts detailed information through a dedicated reference branch. Additionally, Consistency-Regularized Training is introduced to enhance robustness by training the model on pairs of reference images with slight variations in camera pose and lighting, enforcing the generation of lighting-invariant PBR maps. To address misalignment issues between the albedo and MR channels, a Multi-Channel Aligned Attention module is designed to synchronize information across channels, preventing artifacts and unexpected shadows. Furthermore, Learnable Material Embeddings are incorporated for each channel, providing contextual guidance to ensure artifact-free and coherent texture generation.

<p align="left">
  <img src="assets/pipeline.png">
</p>



## 🔗 BibTeX

If you found this repository helpful, please cite our reports:

```bibtex

```