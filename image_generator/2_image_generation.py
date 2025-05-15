from tqdm import tqdm
from diffusers import FluxPipeline
import torch
import numpy as np
from PIL import Image
import os
import json
import shutil
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
sd_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    cache_dir='cache_models/FLUX.1-schnell'
)
sd_pipe.to(device)


def generate_image_by_huric_id(huric_id, config_types, output_folder):
    """
    Generate images for specific huric_id using the Stable Diffusion pipeline.
    """
    huric_id = huric_id.replace("_enriched.hrc", "")
    base_folder = './data'
    config_folder = f'{base_folder}/output_folder/'
    num_images_per_batch_text_to_image = 5
    output_image_width = 1024
    output_image_height = 1024
    num_inference_steps_text_to_image = 10

    shot_types = [
        'close-up shot',
        'wide shot',
        'long shot',
        'low-angle shot',
        'high-angle shot',
    ]
    
    for config_type in config_types:
        configs = []
        for root, _, files in os.walk(config_folder):
            for file in files:
                if file.endswith("_config.json") and huric_id in file and config_type in file:
                    configs.append(os.path.join(root, file))
        for config_path in configs:
            file_name = os.path.basename(config_path)
            case_number = file_name.split('-')[-1].split('_')[0]
            prompts_file_path = f'{config_folder}{file_name.replace("config.json", "response.txt")}'
            
            config = None
            with open(config_path, "r") as json_data:
                config = json.load(json_data)

            prompt_type = config['prompt_type']
            assert prompt_type == config_type, f"\n\nPrompt type mismatch: {prompt_type} != {config_type}"
            
            folder_path = f"{output_folder}/{prompt_type}/{case_number}"
            prompt_list = []
            if os.path.isfile(prompts_file_path):
                try:
                    with open(prompts_file_path, "r") as content_file:
                        prompt_list = eval(content_file.read())
                except Exception as e:
                    print("-----------------EXCEPTION-----------------")
                    print(f"{prompts_file_path=}")
                    print(e)
                    continue
            else:
                print(f"SKIPPING: '{prompts_file_path}' does not exist.")
                continue

            print(f"Processing file: '{prompts_file_path}' ...")
            
            num_batches = len(prompt_list)
            
            for i in tqdm(range(num_batches), 'Prompt Processed', position=1, leave=False):
                prompt = prompt_list[i]
                shot_type = shot_types[i]
                batch = sd_pipe(prompt=prompt,
                                num_images_per_prompt=num_images_per_batch_text_to_image,
                                guidance_scale=3.5,
                                num_inference_steps=num_inference_steps_text_to_image,
                                max_sequence_length=512,
                                generator=torch.Generator("cpu").manual_seed(42),
                                height=output_image_height,
                                width=output_image_width)
                for idx, batch_image in enumerate(batch.images):
                    arr_image = np.array(batch_image)
                    img = Image.fromarray(arr_image)
                    folder_path_with_idx = f"{folder_path}/{shot_type.replace(' ', '_')}/{idx}"
                    os.makedirs(folder_path_with_idx, exist_ok=True)
                    img.save(f"{folder_path_with_idx}/{huric_id}_{prompt_type}_{case_number}_{shot_type.replace(' ', '_')}_{idx}_original.png")
                    with open(f'{folder_path_with_idx}/prompt.json', "w") as output_prompt_file:
                        json.dump({
                            "prompt": prompt,
                            "prompt_type": prompt_type,
                            "shot_type": shot_type
                        }, output_prompt_file)
                    shutil.copy(config_path, folder_path_with_idx)
        

def generate_images():
    """
    Generate images from prompts using the Stable Diffusion pipeline.
    """
    base_folder = './data'
    config_folder = f'{base_folder}/output_folder/'
    output_label_folder_path = f"{base_folder}/output_label_folder/"
    num_images_per_batch_text_to_image = 5
    output_image_width = 1024
    output_image_height = 1024
    num_inference_steps_text_to_image = 10

    shot_types = [
        'close-up shot',
        'wide shot',
        'long shot',
        'low-angle shot',
        'high-angle shot',
    ]

    for config_path in tqdm(glob(f"{config_folder}*_config.json"), 'Huric file processed', position=0):
        file_name = os.path.basename(config_path)
        huric_id = file_name.split('-')[0]
        case_number = file_name.split('-')[-1].split('_')[0]
        prompts_file_path = f'{config_folder}{file_name.replace("config.json", "response.txt")}'
        
        config = None
        with open(config_path, "r") as json_data:
            config = json.load(json_data)

        prompt_type = config['prompt_type']
        
        folder_path = f"{output_label_folder_path}{huric_id}/{prompt_type}/{case_number}"
        if not os.path.exists(folder_path):
            prompt_list = []
            if os.path.isfile(prompts_file_path):
                try:
                    with open(prompts_file_path, "r") as content_file:
                        prompt_list = eval(content_file.read())
                except Exception as e:
                    print("-----------------EXCEPTION-----------------")
                    print(f"{prompts_file_path=}")
                    print(e)
                    continue
            else:
                print(f"SKIPPING: '{prompts_file_path}' does not exist.")
                continue

            print(f"Processing file: '{prompts_file_path}' ...")
            
            num_batches = len(prompt_list)
            
            for i in tqdm(range(num_batches), 'Prompt Processed', position=1, leave=False):
                prompt = prompt_list[i]
                shot_type = shot_types[i]
                batch = sd_pipe(prompt=prompt,
                                num_images_per_prompt=num_images_per_batch_text_to_image,
                                guidance_scale=3.5,
                                num_inference_steps=num_inference_steps_text_to_image,
                                max_sequence_length=512,
                                generator=torch.Generator("cpu").manual_seed(42),
                                height=output_image_height,
                                width=output_image_width)
                for idx, batch_image in enumerate(batch.images):
                    arr_image = np.array(batch_image)
                    img = Image.fromarray(arr_image)
                    folder_path_with_idx = f"{folder_path}/{shot_type.replace(' ', '_')}/{idx}"
                    os.makedirs(folder_path_with_idx, exist_ok=True)
                    img.save(f"{folder_path_with_idx}/{huric_id}_{prompt_type}_{case_number}_{shot_type.replace(' ', '_')}_{idx}_original.png")
                    with open(f'{folder_path_with_idx}/prompt.json', "w") as output_prompt_file:
                        json.dump({
                            "prompt": prompt,
                            "prompt_type": prompt_type,
                            "shot_type": shot_type
                        }, output_prompt_file)
                    shutil.copy(config_path, folder_path_with_idx)
        else:
            print(f"SKIPPING: '{folder_path}' already exists.")
            continue
        
        
if __name__ == "__main__":
    generate_images()