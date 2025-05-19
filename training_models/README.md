
# Fine-Tuning with MiniCPM

This repository provides a full pipeline for fine-tuning the [MiniCPM](https://github.com/OpenBMB/MiniCPM-o) language model on a custom dataset in TSV format. The process includes dataset conversion, optional ID splitting, dataset generation, and training.

## 🔧 Prerequisites

- Install [MiniCPM](https://github.com/OpenBMB/MiniCPM-o) by following the instructions in its official repository.
- Make sure all required dependencies are installed.
- Your dataset should be in `.tsv` format, with appropriate input/output columns. Chose your dataset in image_generator/data/dataset

---

## 📘 Optional: dataset preparation
Below is explained how to prepare the dataset, if you want to skip this step you can use the datasets provided in the folders: 
and go of directly to finetuning the models

### 1 Convert TSV to MiniCPM Format

Use the script `tsv_to_mincpm_format.py` to convert your dataset to MiniCPM's format and add a system prompt to each example. Make sure to change the input_file and output_file variable within the script as desired

```bash
python tsv_to_mincpm_format.py 
````

### 2 (Optional) Create ID Splits for Train/Dev/Test

You can optionally create ID-based splits using `create_id_split.py`. We provide in the repository the split that we used for our experiments called: id_train.txt, id_dev.txt, id_test.txt
Make sure to change the input_path variable within the script as desired
```bash
python create_id_split.py 
```

### 3 Generate Datasets Based on Splits

Now use `generate_dataset.py` to create the final datasets for each split using the ids created with previous step. Make sure to change the filename,directory and directory_target variable within the script as desired

```bash
python generate_splitted_dataset.py
```
---
## ⚙️ Train the Model

Use the provided `.sh` scripts to start training MiniCPM. Before running, make sure to update:
* DATA
* EVAL_DATA
* finetune.py path
* `save_step` depending on the number of examples
* GPU configuration based on your hardware
* deepspeed config path

Example:

```bash
bash finetune_json_top_1.sh
```

## 🧪 Model Evaluation

To evaluate the fine-tuned model, follow these steps:

1. Use the `test.py` script to run inference with the trained model and generate the CSV file required for evaluation.  
   Run the following command:
   ```bash
   python test.py```

2. Use the `evaluate_interpretations.py` script to compute all the relevant evaluation metrics for the model.
   Run the following command:

   ```bash
   python evaluate_interpretations.py
   ```



