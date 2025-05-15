import json
import os
import pandas as pd
import logging

from filter_top_k_images_and_export.huric.parser import parse_huric_file
from filter_top_k_images_and_export.huric.utils import get_hrc_file_path
from filter_top_k_images_and_export.images.bb import compute_frame_semantics_with_bb

logger = logging.getLogger(__name__)

exceptions_images = ["/data/chromei/domestic_image_generator/data/output_label_folder/3084/partial/2/wide_shot/2/3084_partial_2_wide_shot_2_original.png", "/data/chromei/domestic_image_generator/data/output_label_folder/3126/partial/2/wide_shot/4/3126_partial_2_wide_shot_4_original.png", "/data/chromei/domestic_image_generator/data/output_label_folder/3089_20250423/full/0/high-angle_shot/3/3089_full_0_high-angle_shot_3_original.png", "/data/chromei/domestic_image_generator/data/output_label_folder/3089_20250423/full/0/high-angle_shot/1/3089_full_0_high-angle_shot_1_original.png", "/data/chromei/domestic_image_generator/data/output_label_folder/2250/full/3/low-angle_shot/3/2250_full_3_low-angle_shot_3_original.png"]
exceptions_ids = [2433, 3626]

def compute_dataframe(json_file_path, huric_dir, top_k_images, subset_ids: list = []):
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    rows, subset_rows = [], []
    hrc_cache = {}

    for huric_id in json_data.keys():
        if int(huric_id.replace("_enriched", "").replace(".hrc", "")) in exceptions_ids:
            print(f"\t\tSkipping {huric_id} due to exceptions_ids")
            continue

        hrc_file_path = get_hrc_file_path(huric_dir, huric_id)
        try:
            if not os.path.exists(hrc_file_path):
                logger.warning(f"[compute_dataframe] File not found: '{huric_id}' in '{huric_dir}'")
                continue
        except Exception as e:
            logger.error(f"[compute_dataframe] Error accessing file '{huric_id}': {e}")
            continue

        if huric_id in hrc_cache:
            command, frame_semantics, entities, entity_reference_map = hrc_cache[huric_id]
        else:
            command, frame_semantics, entities, entity_reference_map = parse_huric_file(hrc_file_path)
            hrc_cache[huric_id] = (command, frame_semantics, entities, entity_reference_map)

        filtered_images = top_k_images.get(huric_id, [])

        for type_of_prompt, images in filtered_images.items():
            for image_data in images:
                image_path = image_data.get("image", "")

                if image_path in exceptions_images:
                    print(f"\t\tSkipping {image_path} due to exceptions_images")
                    continue

                scores = [float(x) for x in image_data.get("scores", (0, 0))]

                are_visible_entities, are_missing_entities = [], []

                for response in image_data.get("responses", []):
                    entity = response.get("entity_id", "")
                    attribute_name = response.get("attribute", "")
                    question_answer = response.get("answer", "")
                    if attribute_name == "visible":
                        if question_answer == "yes":
                            are_visible_entities.append(entity)
                        else:
                            are_missing_entities.append(entity)

                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                else:
                    image_bytes = None

                obj = {
                    "huric_id": huric_id,
                    "image_path": image_path,
                    "image": image_bytes,
                    "type_of_prompt": type_of_prompt,
                    "visible_entities": ", ".join(entities["visible"]),
                    "are_visible_entities": ", ".join(are_visible_entities),
                    "missing_entities": ", ".join([x for x in entities["missing"] if "person" not in x or "robot" not in x]),
                    "are_missing_entities": ", ".join([x for x in are_missing_entities if "person" not in x or "robot" not in x]),
                    "status_entities": ", ".join(entities["status"]),
                    "near_to_entities": ", ".join(entities["near_to"]),
                    "far_from_entities": ", ".join(entities["far_from"]),
                    "ontop_entities": ", ".join(entities["ontop"]),
                    "inside_entities": ", ".join(entities["inside"]),
                    "command": command,
                    "frame_semantics": frame_semantics,
                    "frame_semantics_bb": "",  # computed later
                    "scores": scores
                }

                try:
                    obj["frame_semantics_bb"] = compute_frame_semantics_with_bb(obj, entity_reference_map, subset_ids)
                except Exception as e:
                    logger.warning(f"[compute_dataframe] Failed to compute frame_semantics_bb for {huric_id}: {e}")
                    obj["frame_semantics_bb"] = "<ERROR>"

                rows.append(obj)

                if int(huric_id.replace("_enriched", "").replace(".hrc", "")) in subset_ids:
                    subset_rows.append(obj)

    df = pd.DataFrame(rows)
    subset_df = pd.DataFrame(subset_rows)
    return df, subset_df
