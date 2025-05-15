# Domestic Image Generator

This README provides a high-level overview of the pipeline used to generate the **Silver Dataset** described in the submitted paper.  

- For general information, refer to the root-level README.  
- For model training details, refer to `training_models/README.md`.

---

## Pipeline Overview

Each step in this pipeline is sequential and corresponds to a specific script or folder:

1. **Constraints Extraction and Prompt Generation** (`prompt_generation/`)
2. **Image Generation** (`2_image_generation.py`)
3. **Image Autolabelling** (`3_autolabeling.py`)
4. **Consistency Checking and Scoring** (`4_consistency_check.py`)
5. **Filtering and Exporting Top-K Images** (`filter_top_k_images_and_export/`)

---

## Installation (via Conda)

### Prerequisites

- CUDA-capable GPU  
- NVIDIA CUDA drivers installed  
- Ensure that `environment.yml` matches your CUDA version (see lines `6`, `14`, and `18`)

### Setup Commands

```bash
export CUDA_HOME=/usr/local/cuda
conda env create -f environment.yml
conda activate visual_grounding
./install_requirements.sh
```

---

## Step 1: Constraints Extraction and Prompt Generation

First, download the dataset files using the instructions in the [HuRIC repository](https://github.com/crux82/huric).

Then edit the path to the HuRIC dataset in `prompt_generation/config.ini`.

Run the following to extract constraints and generate meta-prompts for ChatGPT:

```bash
cd prompt_generation
python main_metaprompt.py --config config.ini
cd ..
```

### Generating Prompts with ChatGPT

#### Manual Method

```bash
mkdir -p data && cp -Ri output_folder data
```

For each file generated, paste the meta-prompt content into [ChatGPT](https://chatgpt.com/) and save the output using the same filename with `_response` appended.  
Example:  
`2175-empty-0.txt` → `2175-empty-0_response.txt`

#### Automatic Method (via OpenAI API)

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Generate prompts:

```bash
python -u generate_prompts.py
```

### Sanity Check

Validate the generated responses:

```bash
python -u response_checker.py
```

This script attempts to auto-correct syntax errors and outputs two diagnostics files:

- `invalid_responses_syntax.txt`: remaining syntax errors (e.g., missing brackets)
- `invalid_responses_elements.txt`: malformed content (e.g., fewer than 5 elements)

**Fix strategy:**

- For syntax issues: manually correct or enhance the script.  
- For malformed content: delete and regenerate the response, or manually complete it.

> **Note**: In our experiments, syntax issues were rare. Element errors were easily fixed via regeneration.

---

## Step 2: Image Generation

Use the prompts to generate images using the [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) diffusion model.

This step is fully automated and skips already-generated images.

```bash
python 2_image_generation.py
```

---

## Step 3: Image Autolabelling

Automatically generate bounding boxes for Existence Constraints using [GroundingDINO](https://github.com/IDEA-Research/Grounded-Segment-Anything.git).

The script loops through images, skipping those already labelled.

```bash
python 3_autolabeling.py
```

---

## Step 4: Consistency Checking and Scoring

Use [MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) to verify **Spatial** and **Status** constraints.  
The script loads HuRIC constraints and queries the Visual LLM, saving all results in a timestamped `.json` file.

```bash
python 4_consistency_check.py
```

---

## Step 5: Filtering and Exporting Top-K Images

To finalise the dataset, filter generated images by score and export the top-k results per HuRIC command.  
If `K = 3`, the script will produce: `top1`, `top2`, and `top3`.

```bash
python -m filter_top_k_images_and_export.main
```

---
