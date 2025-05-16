# Grounded SRL for Human-Robot Interaction

*Grounded Semantic Role Labeling from Synthetic Multimodal Data for Situated Robot Commands*

This project introduces multimodal models for grounded semantic role labeling and the generation of synthetic domestic images. The generated dataset is conditioned on linguistic and environmental constraints extracted from the [HuRIC](https://github.com/crux82/huric) dataset, enabling experiments in **Situated Human-Robot Interaction (HRI)**.

---

## Overview

The repository provides the methods to train and evaluate multimodal models, specifically targeting **Grounded Semantic Role Labelling (G-SRL)** in domestic environments, using synthetic generated images, through a complete pipeline for generating and processing synthetic visual data for robotic command understanding. The pipeline supports:

- Extraction of constraints from HuRIC annotations
- Prompt generation for synthetic image creation
- Image generation using diffusion models
- Automatic bounding box annotation
- Consistency checking with visual LLMs
- Filtering and selection of top-ranked samples

---

## Main Components

This repository includes two primary components:

### 1. `training_models/`
Contains the training and evaluation scripts for applying **MiniCPM-V 2.6** to the **G-SRL** task using the generated and validated dataset.  
Refer to the [README in `training_models/`](training_models/README.md) for configuration and usage.

### 2. `image_generation/`
A self-contained pipeline to create a set of images for the G-SRL dataset. It includes:
- Constraint and prompt generation
- Diffusion-based image generation
- Automatic bounding box labelling
- Visual consistency evaluation
- Top-k image selection

Refer to the [README in `image_generator/`](image_generator/README.md) for full details.


---

## Setup Instructions

The prerequisites and environment setup are common to the entire project. You will need:

- CUDA-capable GPUs
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

1. Running the image generation pipeline (`image_generator/`)
2. Using the generated images to train or evaluate a model (`training_models/`)

If you want to just train the MiniCPM models (Step 2.), you can use our [public available datasets](image_generator/data/dataset) by setting the correct paths.

---

## Documentation

Please refer to the subfolder READMEs for detailed instructions on each component.

---

