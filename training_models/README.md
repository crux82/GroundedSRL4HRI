
# Fine-Tuning with MiniCPM

This repository provides a full pipeline for fine-tuning the [MiniCPM](https://github.com/OpenBMB/MiniCPM-o) language model on a custom dataset in TSV format. The process includes dataset conversion, optional ID splitting, dataset generation, and training.

## 🔧 Prerequisites

- Install [MiniCPM](https://github.com/OpenBMB/MiniCPM-o) by following the instructions in its official repository.
- Make sure all required dependencies are installed.
- Your dataset should be in `.tsv` format, with appropriate input/output columns.

---

## 📘 Step-by-Step Instructions

### 1️⃣ Convert TSV to MiniCPM Format

Use the script `tsv_to_mincpm_format.py` to convert your dataset to MiniCPM's format and add a system prompt to each example. Make sure to change the input_file and output_file variable within the script as desired

```bash
python tsv_to_mincpm_format.py 
````

### 2️⃣ (Optional) Create ID Splits for Train/Dev/Test

You can optionally create ID-based splits using `create_id_split.py`. We provide in the repository the split that we used for our experiments called: id_train.txt, id_dev.txt, id_test.txt
Make sure to change the input_path variable within the script as desired
```bash
python create_id_split.py 
```

### 3️⃣ Generate Datasets Based on Splits

Now use `generate_dataset.py` to create the final datasets for each split using the ids created with previous step. Make sure to change the filename,directory and directory_target variable within the script as desired

```bash
python generate_dataset.py
```

### 4️⃣ Train the Model

Use the provided `.sh` scripts to start training MiniCPM. Before running, make sure to update:
* DATA
* EVAL_DATA
* finetune.py path
* `save_step` depending on the number of examples
* GPU configuration based on your hardware

Example:

```bash
bash train_mycpm.sh
```
