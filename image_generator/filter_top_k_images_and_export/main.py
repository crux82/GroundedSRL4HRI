import json
from heapq import nlargest
import os

from filter_top_k_images_and_export.dataframe.builder import compute_dataframe
from filter_top_k_images_and_export.export.tsv import export_tsv_json_dataset


def sort_and_save(json_file_path):
    """
    Sorts images based on scores[0] and, in case of ties, scores[1].
    """
    result = {}
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    len_images = 0
    for key, value in data.items():
        obj = value['images']
        len_images += len(obj)
        type_groups = {}
        for image_data in obj:
            image_type = image_data["image"].split("/")[-1].split("_")[1]
            if image_type not in type_groups:
                type_groups[image_type] = []
            type_groups[image_type].append(image_data)

        result[key] = {}
        for image_type, images in type_groups.items():
            top_images = sorted(images, key=lambda x: (float(x['scores'][0]), float(x['scores'][1])), reverse=True)
            result[key][image_type] = top_images

    # Save the ordered result to a new JSON file
    output_file_path = json_file_path.replace(".json", "_ordered.json")
    with open(output_file_path, 'w') as file:
        json.dump(result, file, indent=4)
    print(f"Ordered images saved to {output_file_path}")

    print(f"Total number of images: {len_images}")
    print(f"Total number of keys (huric_ids): {len(result)}")
    print(f"Total number of images per key: {len_images / len(result)}")

    return result


def extract_top_k_images(sorted_images, top_k):
    """
    Extracts the top k images based on scores[0] and, in case of ties, scores[1].
    """
    result = {}
    
    for key, value in sorted_images.items():

        result[key] = {}
        for image_type, images in value.items():
            top_images = nlargest(top_k, images, key=lambda x: (float(x['scores'][0]), float(x['scores'][1])))
            result[key][image_type] = top_images

    return result


if __name__ == "__main__":
    json_file_path = "./data/consistency_check_scores_TODAY.json"
    if "TODAY " in json_file_path:
        raise ValueError("Please set the correct path to json_file_path in line 63!\nYou need to replace 'TODAY' with the str representation of the date when you generated the scores, e.g., 20250519.")

    huric_dir = "path/to/huric_dir"  # Replace with the actual path to huric_dir
    if huric_dir.equals("path/to/huric_dir"):
        raise ValueError("Please set the correct path to huric_dir in line 65!.")
    
    sorted_images = sort_and_save(json_file_path)
    
    max_top_k = 10
    for k in range(1, max_top_k+1):
        print(f"EXTRACTING Top_{k} images")
        top_k_images = extract_top_k_images(sorted_images, k)

        output_top_k = json_file_path.replace(".json", f"_top_{k}.json")
        print(f"\tSAVING Top_{k} images to {output_top_k}")
        with open(output_top_k, 'w') as file:
            json.dump(top_k_images, file, indent=4)

        print("\tComputing dataframe...")
        df, subset_df = compute_dataframe(json_file_path, huric_dir, top_k_images)

        tsv_json_file_path = f"./data/dataset/20250514/json_top_{k}_images.tsv"
        os.makedirs(os.path.dirname(tsv_json_file_path), exist_ok=True)
        print(f"\tExporting full dataframe as json in '{tsv_json_file_path}'...")
        export_tsv_json_dataset(df, output_file_path=tsv_json_file_path)
            
    print("\nDone! All exports completed.")