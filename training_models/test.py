import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import json
import argparse
import os
import re
def main(args):
    print("Loading model from:", args.model_type)
    model = AutoModel.from_pretrained(args.model_type, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    processor = AutoProcessor.from_pretrained(args.model_orig, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_orig, trust_remote_code=True)

    with open(args.dataset_path, 'r') as file:
        dataset = json.load(file)

    error = 0
    output_path = args.outputfile
    print(model.generation_config)

    with open(output_path, "w") as file:
        file.write("id;path;command;gold;predicted;is equal\n")

        for item in dataset:
            image_path = item['image']
            target_answer = item['conversations'][-1]['content']
            question = item['conversations'][0]['content'].replace("<image>\n", "")
            print("Processing:", image_path)
     
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")
                continue

            msgs = [{'role': 'user', 'content': [image, question]}]
            answer = model.chat(
                image=None,
                msgs=msgs,
                sampling=False,
                tokenizer=tokenizer
            )
            predicted_answer = answer
            question_filter = re.sub(r".*Now do the same for the following example:\s*", "", question, flags=re.IGNORECASE | re.DOTALL)
            print("Question:", question_filter)
            file.write(f"{item['id']};{image_path};{question_filter};{target_answer};{predicted_answer}\n")

            if predicted_answer != target_answer:
                print("Predicted answer is incorrect.")
                print("ID:", item['id'])
                print("Expected:", target_answer)
                print("Got     :", predicted_answer)
                error += 1
            else:
                print("Correct prediction.")
            print("-" * 50)

    print(f"Total errors: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a dataset with a fine-tuned model.")

    parser.add_argument("--model_type", type=str, required=True,
                        help="Path to the fine-tuned model checkpoint.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset JSON file.")
    parser.add_argument("--model_orig", type=str, default="openbmb/MiniCPM-V-2_6",
                        help="Base model type for processor and tokenizer.")
    parser.add_argument("--outputfile", type=str, required=True,
                        help="Output file path to save predictions.")

    args = parser.parse_args()
    main(args)

