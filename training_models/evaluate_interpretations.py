# pip install numpy==1.24.4
# pip install pandas==2.0.3
# pip install json5==0.12.0
# pip install scikit-learn==1.3.2
# pip install scipy==1.10.0

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Tuple, Optional
import argparse
import pandas as pd
import json
import json5
import os
import re
from collections import Counter
from itertools import zip_longest
import ast


def attempt_json_recovery(s: str) -> str:
    s = s.strip()

    # 1. Replace single quotes with double quotes (naive but effective in model generations)
    s = re.sub(r"(?<!\\)'", '"', s)

    # 2. Remove trailing commas before closing braces/brackets
    s = re.sub(r",\s*([\]}])", r"\1", s)

    # 3. Balance brackets and braces
    count_open = s.count('{')
    count_close = s.count('}')
    if count_close < count_open:
        s += '}' * (count_open - count_close)
    elif count_open < count_close:
        s = s.rstrip('}')  # just truncate extra closing braces

    count_open = s.count('[')
    count_close = s.count(']')
    if count_close < count_open:
        s += ']' * (count_open - count_close)
    elif count_open < count_close:
        s = s.rstrip(']')

    return s


def normalize_surface(s: str) -> str:
    return ''.join(s.lower().split())


# This function interprets a string containing a list of numbers and returns a list of integers.
# It handles the case where the input contains numbers with leading zeros.
# For example, "001" will be converted to 1.
def parse_bounding_box(bbox) -> List[int]:

    # Check if the input is a string representation of a list
    if isinstance(bbox, str):
        # Remove leading and trailing spaces
        bbox = bbox.strip()
        # Remove brackets and split by comma
        bbox = bbox.strip('[]').split(',')
    elif isinstance(bbox, list):
        # If it's already a list, just use it directly
        pass
    else:
        raise ValueError("Input must be a string or list.")

    # Convert each element to an integer, removing leading zeros
    return [int(el.strip()) for el in bbox]


def iou(box1, box2) -> float:
    try:
        x1, y1, x2, y2 = parse_bounding_box(box1)
        x1_p, y1_p, x2_p, y2_p = parse_bounding_box(box2)
    except Exception as e:    
        print(f"Error parsing bounding boxes: {box1}, {box2}. Error: {e}")
        quit()

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def match_frames(gold_frames: List[Dict[str, Any]], pred_frames: List[Dict[str, Any]]) -> Tuple:
    n_gold = len(gold_frames)
    n_pred = len(pred_frames)
    cost_matrix = np.zeros((n_gold, n_pred))

    for i, gf in enumerate(gold_frames):
        for j, pf in enumerate(pred_frames):
            score = 0
            if gf['frame'] == pf['frame']:
                score += 1

            gold_fes = {el['name']: el for el in gf['elements']}
            pred_fes = {el['name']: el for el in pf['elements']}
            common_keys = gold_fes.keys() & pred_fes.keys()
            score += len(common_keys)
            print("g",gold_fes)
            print("p",pred_fes)
            for k in common_keys:
                gs = normalize_surface(gold_fes[k]['surface'])
                ps = normalize_surface(pred_fes[k]['surface'])
                if gs == ps:
                    score += 1

            cost_matrix[i, j] = -score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, cost_matrix


def compute_counts_multiset(true_list: List[str], pred_list: List[str]) -> Tuple[int, int, int]:
    true_counter = Counter(true_list)
    pred_counter = Counter(pred_list)
    tp = sum((true_counter & pred_counter).values())
    fp = sum((pred_counter - true_counter).values())
    fn = sum((true_counter - pred_counter).values())
    return tp, fp, fn


def evaluate_semantic_interpretations_strict(gold_list: List[List[Dict[str, Any]]],
                                              pred_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    total_frame_tp = total_frame_fp = total_frame_fn = 0
    total_fe_tp = total_fe_fp = total_fe_fn = 0
    total_surf_tp = total_surf_fp = total_surf_fn = 0
    total_tags_tp = total_tags_fp = total_tags_fn = 0

    iou_scores = []
    iou_scores_match_only = []
    iou_accuracy = []
    bb_tag_matches = []
    iou_threshold = 0.5
    

    for idx, el in enumerate(zip(gold_list, pred_list)):
        gold_frames, pred_frames = el
        row_ind, col_ind, _ = match_frames(gold_frames, pred_frames)

        gold_frame_list = [f['frame'] for f in gold_frames]
        pred_frame_list = [f['frame'] for f in pred_frames]
        tp, fp, fn = compute_counts_multiset(gold_frame_list, pred_frame_list)
        
        total_frame_tp += tp
        total_frame_fp += fp
        total_frame_fn += fn

        local_fe_gold = []
        local_fe_pred = []

        local_surf_gold = []
        local_surf_pred = []
        
        local_bbox_gold = []
        local_bbox_pred = []

        local_tags_gold = []
        local_tags_pred = []

        for i, j in zip(row_ind, col_ind):
            g_frame = gold_frames[i]
            p_frame = pred_frames[j]
            g_frame_name = g_frame['frame']
            p_frame_name = p_frame['frame']

            g_fes = {el['name']: el for el in g_frame['elements']}
            p_fes = {el['name']: el for el in p_frame['elements']}
            all_keys = g_fes.keys() | p_fes.keys()
            
            for key in all_keys:
                if key in g_fes:
                    local_fe_gold.append(f"{g_frame_name}:{key}")
                    g_surf = normalize_surface(g_fes[key]['surface'])
                    local_surf_gold.append(f"{g_frame_name}:{key}:{g_surf}")
                    local_tags_gold.append(g_fes[key]['bbox_2d'] if isinstance(g_fes[key]['bbox_2d'], str) else "<BBOX_2D>")
                if key in p_fes:
                    local_fe_pred.append(f"{p_frame_name}:{key}")
                    p_surf = normalize_surface(p_fes[key]['surface'])
                    local_surf_pred.append(f"{p_frame_name}:{key}:{p_surf}")
                    local_tags_pred.append(p_fes[key]['bbox_2d'] if isinstance(p_fes[key]['bbox_2d'], str) else "<BBOX_2D>")

        # loop through the gold frames
        for i, g_frame in enumerate(gold_frames):
            for g_el in g_frame['elements']:
                local_bbox_gold.append(f"{g_el['surface']}:{str(g_el['bbox_2d']).replace(')', '').replace('(', '')}")
        # loop through the pred frames
        for i, f_frame in enumerate(pred_frames):
            for f_el in f_frame['elements']:
                local_bbox_pred.append(f"{f_el['surface']}:{str(f_el['bbox_2d']).replace(')', '').replace('(', '')}")
        
        
        for g_bbox_str, p_bbox_str in zip_longest(local_bbox_gold, local_bbox_pred, fillvalue="None:None"):
            _, g_bbox = g_bbox_str.split(':')
            _, p_bbox = p_bbox_str.split(':')
            print("##")
            print(g_bbox)
            print(p_bbox)           
            if g_bbox.startswith("[") and p_bbox.startswith("["):

                score = iou(g_bbox, p_bbox)
                print(score)
                iou_scores.append(score)
                iou_accuracy.append(score > iou_threshold)
                iou_scores_match_only.append(score)
            elif g_bbox.startswith("[") and not p_bbox.startswith("["):
                iou_scores.append(0.0)
                iou_accuracy.append(False)
                bb_tag_matches.append(False)
            elif g_bbox.startswith("<") and p_bbox.startswith("<"):
                bb_tag_matches.append(g_bbox_str == p_bbox_str)
            else:
                bb_tag_matches.append(False)
        
        fe_tp, fe_fp, fe_fn = compute_counts_multiset(local_fe_gold, local_fe_pred)
        total_fe_tp += fe_tp
        total_fe_fp += fe_fp
        total_fe_fn += fe_fn

        surf_tp, surf_fp, surf_fn = compute_counts_multiset(local_surf_gold, local_surf_pred)
        total_surf_tp += surf_tp
        total_surf_fp += surf_fp
        total_surf_fn += surf_fn

        tags_tp, tags_fp, tags_fn = compute_counts_multiset(local_tags_gold, local_tags_pred)
        total_tags_tp += tags_tp
        total_tags_fp += tags_fp
        total_tags_fn += tags_fn
    
        # if idx % 2 == 0 and idx > 10:
        #     quit()

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return {"Precision": prec, "Recall": rec, "F1": f1}

    return {
        "Frame Match": prf(total_frame_tp, total_frame_fp, total_frame_fn),
        "FE Match": prf(total_fe_tp, total_fe_fp, total_fe_fn),
        "Surface Match": prf(total_surf_tp, total_surf_fp, total_surf_fn),
        "Bounding Box IoU Mean": float(np.mean(iou_scores)) if iou_scores else None,
        "Bounding Box IoU Mean (Match Only)": float(np.mean(iou_scores_match_only)) if iou_scores_match_only else None,
        "Bounding Box IoU Mean Accuracy (IoU > 0.5)": float(np.mean(iou_accuracy)) if iou_accuracy else None,
        "Bounding Box Tag Match": float(np.mean(bb_tag_matches)) if bb_tag_matches else None,
        "TAGS P/R/F1": prf(total_tags_tp, total_tags_fp, total_tags_fn)
    }


def safe_json_load(s: str, fallback: Any = [], log_file: Optional[str] = None, index: Optional[int] = None) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        corrected = attempt_json_recovery(s)
        try:
            return json.loads(corrected)
        except json.JSONDecodeError:
            try:
                return json5.loads(s)
            except Exception as e:
                warning = f"[WARNING] Row {index} Failed all recovery attempts.\nOriginal: {s}\nRecovery failed: {e}\n"
                print(warning)
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(warning + '\n')
                return fallback


# EXAMPLE of input interpretation:
# "INSPECTING(Ground(closet, [.., .., .., ..])) - BEING_LOCATED(Theme(that, <POSITION>), Location(bedroom, <ROOM>))"
def get_json_interpretation(interpretation):
    #print(interpretation)
    """
    Construct a JSON object from the interpretation string, i.e. a list where each Frame is an object
    containing the Frame name and its respective Frame Elements. Each Frame Element contains the Element name and its respective
    fillers and bounding boxes.
    """
    
    json_interpretation = []
    
    for frame_string in interpretation.split(" - "):
        frame_name = frame_string.split("(")[0].strip()
        frame_obj = {"frame": frame_name, "elements": []}
        frame_elements = "(".join(frame_string.split("(")[1:]).split("),")
        for frame_element in frame_elements:
            frame_element_name = frame_element.split("(")[0].strip()
            if "(" not in frame_element:
                continue
            try:
                frame_element_value = frame_element.split("(")[1].split(",")[0].strip().lower()
                frame_element_bb = frame_element.split("(")[1].split(",")[1].strip().replace(")", "")
                if frame_element_bb.startswith("["):
                    frame_element_bb = ", ".join(frame_element.split("(")[1].split(",")[1:]).strip().replace(")", "")
            except IndexError:
                print(f"IndexError: {frame_element}")
                return {}
            
            frame_element_obj = {
                "name": frame_element_name,
                "surface": frame_element_value,
                "bbox_2d": frame_element_bb
            }
            
            frame_obj["elements"].append(frame_element_obj)
            
        json_interpretation.append(frame_obj)
        
    return json_interpretation


def load_interpretations_from_csv(csv_path: str, log_path: str) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    df = pd.read_csv(csv_path, sep=";")
    #print(df)
    file_type = csv_path.split("/")[-1].split("_")[1]
    gold_list = []
    pred_list = []
    for idx, row in df.iterrows():
        # TODO: change here name of rows if different!
        if file_type == "text":
            gold = get_json_interpretation(row['gold'])
            pred = get_json_interpretation(row['predicted'])
        elif file_type == "json":
            gold = safe_json_load(row['gold'], log_file=log_path, index=idx)
            pred = safe_json_load(row['predicted'], log_file=log_path, index=idx)
        gold_list.append(gold)
        pred_list.append(pred)
    return gold_list, pred_list


def test():
    cases = [
        {
            "description": "Perfect match",
            "gold": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'apple', 'bbox_2d': [1,1,5,5]}]}]],
            "pred": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'apple', 'bbox_2d': [1,1,5,5]}]}]],
            "expected": {
                "Frame Match": 1.0,
                "FE Match": 1.0,
                "Surface Match": 1.0,
                "IOU Match": 1.0
            }
        },
        {
            "description": "Wrong frame name",
            "gold": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'apple', 'bbox_2d': [0,0,5,5]}]}]],
            "pred": [[{'frame': 'TAKING', 'elements': [{'name': 'Theme', 'surface': 'apple', 'bbox_2d': [0,0,5,5]}]}]],
            "expected": {
                "Frame Match": 0.0
            }
        },
        {
            "description": "Different FE name",
            "gold": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'banana', 'bbox_2d': '<MISSING>'}]}]],
            "pred": [[{'frame': 'BRINGING', 'elements': [{'name': 'Item', 'surface': 'banana', 'bbox_2d': '<MISSING>'}]}]],
            "expected": {
                "FE Match": 0.0,
                "Surface Match": 0.0,
                "IOU Match": 1.0
            }
        },
        {
            "description": "BBox mismatch",
            "gold": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'pen', 'bbox_2d': [0,0,10,10]}]}]],
            "pred": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'pen', 'bbox_2d': [8,8,18,18]}]}]],
            "expected": {
                "IOU Match": 0.0
            }
        },
        {
            "description": "Surface with case and spaces",
            "gold": [[{'frame': 'BRINGING', 'elements': [{'name': 'Goal', 'surface': '   Kitchen ', 'bbox_2d': '<ROOM>'}]}]],
            "pred": [[{'frame': 'BRINGING', 'elements': [{'name': 'Goal', 'surface': 'kitchen', 'bbox_2d': '<ROOM>'}]}]],
            "expected": {
                "Surface Match": 1.0
            }
        },
        {
            "description": "Multiple frames with partial match",
            "gold": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'newspaper', 'bbox_2d': [329, 641, 795, 778]}, {'name': 'Goal', 'surface': 'studio', 'bbox_2d': '<ROOM>'}]}, {'frame': 'BRINGING','elements': [{'name': 'Theme', 'surface': 'bottle', 'bbox_2d': "<MISSING>"}, {'name': 'Goal', 'surface': 'studio', 'bbox_2d': '<ROOM>'}]}], [{'frame': 'BRINGING','elements': [{'name': 'Theme', 'surface': 'bottle', 'bbox_2d': "<MISSING>"}, {'name': 'Goal', 'surface': 'studio', 'bbox_2d': '<ROOM>'}]}]],
            "pred": [[{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'Newspaper ', 'bbox_2d': [330, 640, 794, 780]}, {'name': 'Goal', 'surface': 'Studio', 'bbox_2d': '<ROOM>'}]}], [{'frame': 'BRINGING', 'elements': [{'name': 'Theme', 'surface': 'bottle', 'bbox_2d': '<MISSING>'}, {'name': 'Goal', 'surface': 'studio', 'bbox_2d': '<ROOM>'}]}]],
            "expected": {
                "Frame Match": 0.8,
                "FE Match": 1.0,
                "Surface Match": 1.0,
                "IOU Match": 1.0
            }
        }
    ]

    for i, case in enumerate(cases):
        print(f"\nTest {i+1}: {case['description']}")
        result = evaluate_semantic_interpretations_strict(case['gold'], case['pred'])
        # print results and V if value['F1'] == case['expected'] else X
        for key, value in result.items():
            print(key)
            print(value)
            print(50*"*")
            # if key == "Bounding Box Match Rate (>0.5 IOU or tag match)":
            #     print(f"\t{key} \n\t\t Expected: {case['expected'].get('IOU Match', 'N/A')}, Got: {value}", "\t", "✅" if value == case['expected'].get('IOU Match', 'N/A') else "" if case['expected'].get('IOU Match', 'N/A') == 'N/A' else "🔴")
            # elif key == "Bounding Box IOU Mean":
            #     print(f"\t{key} \n\t\t Expected: {case['expected'].get('Bounding Box IOU Mean', 'N/A')}, Got: {value}", "\t", "✅" if value == case['expected'].get('Bounding Box IOU Mean', 'N/A') else "" if case['expected'].get('Bounding Box IOU Mean', 'N/A') == 'N/A' else "🔴")
            # else:
            #     print(f"\t{key} \n\t\t Expected: {case['expected'].get(key, 'N/A')}, Got: {value}", "\t", "✅" if value['F1'] == case['expected'].get(key, 'N/A') else "" if case['expected'].get(key, 'N/A') == 'N/A' else "🔴")
        
    print("Test completed.")



import csv
def main():
    parser = argparse.ArgumentParser(description="Evaluate Frame Semantics Interpretations")
    parser.add_argument('--input', type=str, required=True, help='Path to CSV file with columns: GS and PRED')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output JSON with metrics')
    args = parser.parse_args()

    log_path = os.path.splitext(args.output)[0] + "_parsing_warnings.txt"
    gold_list, pred_list = load_interpretations_from_csv(args.input, log_path)
    results = evaluate_semantic_interpretations_strict(gold_list, pred_list)
    

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    headers = []
    values = []

    for key, value in results.items():
       if isinstance(value, dict):
           for subkey, subval in value.items():
               headers.append(f"{key} - {subkey}")
               values.append(subval)
       else:
           headers.append(key)
           values.append(value)

    # Scrittura su file CSV con separatore ";"
    with open(args.output, "w", newline="") as f:
       writer = csv.writer(f, delimiter=';')
       writer.writerow(headers)
       writer.writerow(values)
    print(f"Evaluation completed and saved to: '{args.output}'.\nWarnings (if any) saved to: '{log_path}'")
    
    '''
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation completed and saved to: '{args.output}'.\nWarnings (if any) saved to: '{log_path}'")
    '''


if __name__ == '__main__':
    main()
    # test()
    