import os
import logging
from datetime import datetime
from prompt_generation.main_metaprompt import generate_metaprompts_for_huric_ids
from prompt_generation.generate_prompts import generate_prompts
from prompt_generation.response_checker import check_response_syntax

# module starts with numerical character, must be imported with __import__
generate_image_by_huric_id = __import__("0_image_generation").generate_image_by_huric_id
autolabeling_for_ids = __import__("1_autolabeling").autolabeling_for_ids
process_huric_ids = __import__("2_consistency_check").process_huric_ids
    
# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s][%(filename)s][%(funcName)s()] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Paths
THRESHOLD_FILE = "./data/below_threshold_list_2025_04_26.txt"
# THRESHOLD_FILE = "./data/below_threshold_list_TEST.txt"
IMAGES_ROOT = "./data/output_label_folder"
OUTPUT_ROOT = "./data/output_folder"

def read_below_threshold_list(path):
    """
    Read lines formatted as id_huric;configuration with no header.
    Returns list of (id_huric, configuration).
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                id_huric, config = line.split(";")
                id_huric = id_huric.strip().replace("_enriched", "").replace(".hrc", "")
                entries.append((id_huric, config))
            except ValueError:
                logging.warning(f"Skipping malformed line: {line}")
    
    # make entries a list of tuples where the first element is the id_huric and the second is a list of configurations
    # since the id_huric is repeated in the file with different configurations, we must merge them into a single list 
    # together with the id_huric
    merged_entries = {}
    for id_huric, config in entries:
        if id_huric not in merged_entries:
            merged_entries[id_huric] = []
        merged_entries[id_huric].append(config)
    
    # convert the merged_entries dictionary to a list of tuples
    entries = [(id_huric, configs) for id_huric, configs in merged_entries.items()]
    
    return entries

def make_new_folder(output_root, id_huric):
    """
    Create a new folder named <id_huric>_<YYYYMMDD> under output_root.
    Returns the new folder path.
    """
    id_huric = id_huric.replace("_enriched", "").replace(".hrc", "")
    date_str = datetime.now().strftime("%Y%m%d")
    new_name = f"{id_huric}_{date_str}"
    new_path = os.path.join(output_root, new_name)
    os.makedirs(new_path, exist_ok=True)
    logging.info(f"Created or found folder: {new_path}")
    return new_path


def delete_prompt_response(output_root, id_huric, configs: list):
    """
    Deletes the '_response.txt' file inside output_root
    """
    # remove "_enriched" and ".hrc" from id_huric to match the file naming convention
    id_huric = id_huric.replace("_enriched", "").replace(".hrc", "")
    
    # first find all files that end with "_response.txt" and contain both id_huric and config
    for config in configs:
        for file in os.listdir(output_root):
            if file.endswith("_response.txt") and id_huric in file and config in file:
                # then delete them
                file_path = os.path.join(output_root, file)
                logging.info(f"Deleting {file_path}")
                os.remove(file_path)

def call_prompts_generation(tasks):
    logging.info(f"Regenerating prompts")
    # logging.info("generate_metaprompts_for_huric_ids FUNCTION IS COMMENTED OUT")
    generate_metaprompts_for_huric_ids([id_huric for id_huric, _, _ in tasks], [configs for _, configs, _ in tasks])
    # call here the prompt generation module
    generate_prompts(input_directory=OUTPUT_ROOT)
    # check the syntax of the response files
    check_response_syntax(input_directory=OUTPUT_ROOT)
    

def call_image_generation(tasks):
    """
    Calls the image_generation process for each task.
    Each task is a tuple of (id_huric, config_types, folder).
    """
    logging.info(f"Call image_generation for {len(tasks)} tasks")
    for id_huric, config_types, folder in tasks:
        logging.info(f"  -> Generate images for {id_huric} in {folder}... with {len(config_types)} configurations")
        generate_image_by_huric_id(id_huric, config_types, folder)


def call_autolabeling(ids_list):
    """
    Calls the autolabeling process for images of the given folders.
    """
    logging.info(f"Call the autolabeling process for {len(ids_list)} files")
    autolabeling_for_ids(ids_list)
    

def call_consistency_check(id_list):
    """
    Calls the consistency_check process for the given list of ids.
    """
    logging.info(f"Call consistency_check for {len(id_list)} files")
    process_huric_ids(id_list)


def main():
    logging.info("Starting backup process")

    # 1. Read the below-threshold list
    below_threshold = read_below_threshold_list(THRESHOLD_FILE)
    if not below_threshold:
        logging.info("No files below threshold. Exiting.")
        return

    # 2. For each failing config, prepare a new dated folder and collect tasks
    regen_tasks = []
    ids_to_check = set()

    for id_huric, configs in below_threshold:
        # first delete the old response file
        # logging.info("DELETE FUNCTION IS COMMENTED OUT")
        delete_prompt_response(OUTPUT_ROOT, id_huric, configs)
        # then create a new folder for the images that will be generated
        new_images_folder = make_new_folder(IMAGES_ROOT, id_huric)
        regen_tasks.append((id_huric, configs, new_images_folder))
        ids_to_check.add(id_huric)
    
    # logging.info(f"{regen_tasks=}")
    # logging.info(f"{ids_to_check=}")

    # 3. Regenerate prompts and assemble image-generation tasks
    call_prompts_generation(regen_tasks)

    # 4. Call the image_generation process
    call_image_generation(regen_tasks)
    
    # 5. Call the autolabeling process
    call_autolabeling(list(ids_to_check))

    # 6. Call consistency_check process
    call_consistency_check(list(ids_to_check))

    logging.info("Backup process completed")


if __name__ == "__main__":
    main()
