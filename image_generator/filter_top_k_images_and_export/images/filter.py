import json
import os
import logging
from heapq import nlargest
from datetime import datetime

THRESHOLD = 0.5

TODAY = datetime.today().strftime('%Y_%m_%d')
below_threshold_filepath = f"/data/chromei/domestic_image_generator/data/below_threshold_list_{TODAY}.txt"

logger = logging.getLogger(__name__)

def filter_top_k_images(json_file_path, top_k, verbose: bool = False):
    result = {}
    below_threshold_list = []

    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for key, value in data.items():
        obj = value['images']
        type_groups = {}
        for image_data in obj:
            image_type = image_data["image"].split("/")[-1].split("_")[1]
            if image_type not in type_groups:
                type_groups[image_type] = []
            type_groups[image_type].append(image_data)

        result[key] = {}
        for image_type, images in type_groups.items():
            top_images = nlargest(top_k, images, key=lambda x: (float(x['scores'][0]), float(x['scores'][1])))
            result[key][image_type] = top_images

            below_threshold = True
            for image_data in top_images:
                scores = image_data.get("scores", None)
                if scores[0] >= THRESHOLD or scores[1] >= THRESHOLD:
                    below_threshold = False
                    break
            
            if below_threshold:
                below_threshold_list.append([key, image_type])
                if verbose:
                    logger.warning(f"Scores for huric_id='{key} - {image_type}' are below the threshold.")

    os.makedirs(os.path.dirname(below_threshold_filepath), exist_ok=True)
    with open(below_threshold_filepath, 'w') as file:
        for item in below_threshold_list:
            file.write(f"{item[0]};{item[1]}\n")
     
    return result
