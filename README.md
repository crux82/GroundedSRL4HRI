# Domestic Image Generator

*Grounded Semantic Role Labeling from Synthetic Multimodal Data for Situated Robot Commands*

This project supports the generation of synthetic domestic images and their use in training multimodal models for grounded semantic role labeling. The generated dataset is conditioned on linguistic and environmental constraints extracted from the [HuRIC](https://github.com/crux82/huric) dataset, enabling experiments in **Situated Human-Robot Interaction (HRI)**.

---

## Overview

The repository provides a complete pipeline for generating and processing synthetic visual data for robotic command understanding. The pipeline supports:

- Extraction of constraints from HuRIC annotations
- Prompt generation for synthetic image creation
- Image generation using diffusion models
- Automatic bounding box annotation
- Consistency checking with visual LLMs
- Filtering and selection of top-ranked samples

These synthetic images are then used to train and evaluate multimodal models, specifically targeting **Grounded Semantic Role Labelling (G-SRL)** in domestic environments.

---

## Main Components

This repository includes two primary components:

### 1. `image_generation/`
A self-contained pipeline to create the Silver dataset. It includes:
- Constraint and prompt generation
- Diffusion-based image generation
- Automatic bounding box labelling
- Visual consistency evaluation
- Top-k image selection

Refer to the [README in `image_generation/`](image_generation/README.md) for full details.

### 2. `training_models/`
Contains the training and evaluation scripts for applying **MiniCPM-V 2.6** to the **G-SRL** task using the generated Silver dataset.  
Refer to the [README in `training_models/`](training_models/README.md) for configuration and usage.

---

## Setup Instructions

The prerequisites and environment setup are common to the entire project. You will need:

- CUDA-capable GPU
- NVIDIA CUDA drivers installed
- Python + Conda

Create the environment as follows:

```bash
export CUDA_HOME=/usr/local/cuda
conda env create -f environment.yml
conda activate visual_grounding
./install_requirements.sh
```

---

## Getting Started

Each subfolder includes a dedicated README to walk you through its functionality. A typical workflow consists of:

1. Running the image generation pipeline (`image_generation/`)
2. Using the generated images to train or evaluate a model (`training_models/`)

---

## Documentation

Please refer to the subfolder READMEs for detailed instructions on each component.

---

