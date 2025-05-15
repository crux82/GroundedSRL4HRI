from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import os, sys
import torch
import json
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict
from huggingface_hub import hf_hub_download


base_folder = './data'
output_label_folder_path = f"{base_folder}/output_label_folder/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['SUPERVISON_DEPRECATION_WARNING'] = '0'
grounding_dino_path = os.path.join(os.getcwd(), "GroundingDINO")
sys.path.append(grounding_dino_path)

TODAY = datetime.now().strftime("%Y%m%d")


# Helper functions
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, 
                                    filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), 
                                                        strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


# loading model 
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, 
                                    ckpt_filename, 
                                    ckpt_config_filename, 
                                    device)
sys.path.remove(grounding_dino_path)



# detect object using grounding DINO
def detect(image_source, image_preprocessed, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image_preprocessed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, 
                                    boxes=boxes, 
                                    logits=logits,
                                    phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    return annotated_frame, boxes, phrases, logits

# image preprocess for groundingDINO model
def preprocess_image(pil_image):
    image = np.array(pil_image)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(pil_image, None)
    return image, image_transformed


def area_bounding_box(bbox):
    top, left, bottom, right = bbox
    return (bottom - top) * (right - left)
    
def is_bbox_larger(bbox_1, bbox_2):
    if bbox_2 is None:
        return True
    area_1 = area_bounding_box(bbox_1)
    area_2 = area_bounding_box(bbox_2)
    return area_1 > area_2

def scoring_function(entities_to_include, entities_to_exclude, single_image_annotations):
    prefix_removed_single_image_annotations = [(bbox, label[2:], pred_score) for bbox, label, pred_score in single_image_annotations]
    lower_entities_to_include = [[e['type'].lower(), e['atom']] for e in entities_to_include]
    lower_entities_to_exclude = [[e['type'].lower(), e['atom']] for e in entities_to_exclude] + [['person', 'person'], ['robot', 'robot']] # type and atom
    entities_to_be_excluded_found = []
    entities_to_be_included_missing = []
    labeled_bboxes = []
    eps = 1e-3
    score = 1 if any(lower_entities_to_include) or any(lower_entities_to_exclude) else 1 - eps
    for entity_to_include, atom in lower_entities_to_include:
        best_selected = None
        best_score = 0
        best_bbox = None
        for bbox, label, pred_score in prefix_removed_single_image_annotations:
            if entity_to_include == label.lower() and (pred_score > best_score or (pred_score == best_score and is_bbox_larger(bbox, best_bbox))):
                best_selected = (bbox, label.lower(), pred_score)
                best_score = pred_score
                best_bbox = bbox
        if best_selected is not None:
            bbox, label, pred_score = best_selected
            labeled_bboxes.append([bbox, atom, pred_score])
            score *= float(pred_score)
        else:
            score *= eps
            entities_to_be_included_missing.append(entity_to_include)

    for entity_to_exclude, atom in lower_entities_to_exclude:
        best_selected = None
        best_score = 0
        best_bbox = None
        for bbox, label, pred_score in prefix_removed_single_image_annotations:
            if entity_to_exclude == label.lower() and (pred_score > best_score or (pred_score == best_score and is_bbox_larger(bbox, best_bbox))):
                best_selected = (bbox, label.lower(), pred_score)
                best_score = pred_score
                best_bbox = bbox
        if best_selected is not None:
            bbox, label, pred_score = best_selected
            score *= 1 - float(pred_score)
            labeled_bboxes.append([bbox, atom, pred_score])
            entities_to_be_excluded_found.append(label)
        else:
            score *= 1 - eps
    
    return score, entities_to_be_included_missing, entities_to_be_excluded_found, labeled_bboxes


def at_least_one_element_in_common(a, b):
    return set(a) & set(b)


def autolabeling_for_ids(ids_list):
    images_paths = []
    labeled_images_paths = []
    scores = []
    bboxes = []
    config_files = []
    commands = []
    huric_ids = []
    prompts = []
    prompt_types = []
    important_entities_to_include_list = []
    important_entities_to_exclude_list = []
    optional_entities_list = []
    frame_semantics_list = []
    spatial_relations_list = []
    excluded_entities_found_indicators = []
    important_entity_not_found_indicators = []
    shot_types = []
    negative_example_bool_list = []
    negative_example_entities_list = []
    negative_example_entities_count_list = []
    labeled_bboxes_list = []


    for image_path in tqdm(glob(f'{output_label_folder_path}**/*_original.png', recursive=True)):
        image_file_name = os.path.basename(image_path)
        current_folder = os.path.dirname(image_path)
        if image_file_name.split('_')[0] in ids_list: 
            if TODAY in image_path:
                annotated_image_path = f"{current_folder}/{image_file_name.replace('original', 'annotated')}"

                if os.path.isfile(annotated_image_path):
                    print(f"Skipping '{annotated_image_path}' as it is already processed. Check if it's present in the 'dataset.csv' file.")
                    continue

                config_path = glob(f'{current_folder}/*_config.json')[0]
                with open(config_path, "r", encoding="utf8") as json_data:
                    config = json.load(json_data)

                pil_image = Image.open(f'{image_path}')

                with open(f"{current_folder}/prompt.json", "r", encoding="utf8") as prompt_file:
                    prompt_file_content = json.load(prompt_file)
                    prompt = prompt_file_content['prompt']
                    shot_type = prompt_file_content["shot_type"]

                huric_id = image_file_name.split('_')[0]
                command = config['command']
                entities_to_include = config['important_entities_to_include']
                entities_to_include_names = [e['type'] for e in entities_to_include]
                entities_to_include_ids = [e['atom'] for e in entities_to_include]
                entities_to_exclude = config['important_entities_to_exclude'] 
                entities_to_exclude_names = [e['type'] for e in entities_to_exclude] + ['person', 'robot']
                entities_to_exclude_ids = [e['atom'] for e in entities_to_exclude] + ['person_123456789', 'robot_123456789']
                optional_entities = config['optional_entities']
                frame_semantics = config['frame_semantics']
                spatial_relations = config['spatial_relations']
                prompt_type = config['prompt_type']
                negative_example_bool = config['negative_example'] 
                negative_example_entities = config['negative_example_entities'] 
                negative_example_entities_count = config['negative_example_entities_count'] 
                
                important_entities_to_include_list.append(entities_to_include)
                important_entities_to_exclude_list.append(entities_to_exclude)
                optional_entities_list.append(optional_entities)
                frame_semantics_list.append(frame_semantics)
                spatial_relations_list.append(spatial_relations)

                entity_names = entities_to_include_names + entities_to_exclude_names
                entity_ids = entities_to_include_ids + entities_to_exclude_ids

                image_source, image_preprocessed = preprocess_image(pil_image)
                annotated_frame, detected_bboxes, phrases, logits = detect(image_source, 
                                                        image_preprocessed,
                                                        text_prompt=" ".join(["a " + e.lower() + '.' for e in entity_names]), 
                                                        model=groundingdino_model,
                                                        box_threshold = 0.4,
                                                        text_threshold = 0.3)
                
                annotations = list(zip(detected_bboxes, phrases, logits))
                json_annotations = list(zip(detected_bboxes.numpy().tolist(), phrases, entity_ids, logits.numpy().tolist()))

                with open(f'{current_folder}/annotations.txt', 'w') as f:
                    json.dump(json_annotations, f)

                Image.fromarray(annotated_frame).save(annotated_image_path)

                score, important_entities_missing, entities_to_be_excluded_found, labeled_bboxes = scoring_function(entities_to_include, entities_to_exclude, annotations)

                scores.append(score)
                labeled_images_paths.append(annotated_image_path)
                bboxes.append(detected_bboxes.numpy().tolist())
                images_paths.append(image_path)
                config_files.append(config_path)
                commands.append(command)
                huric_ids.append(huric_id)
                prompts.append(prompt)
                prompt_types.append(prompt_type)
                important_entity_not_found_indicators.append(important_entities_missing)
                excluded_entities_found_indicators.append(entities_to_be_excluded_found)
                shot_types.append(shot_type)
                labeled_bboxes_list.append(labeled_bboxes)
                negative_example_bool_list.append(negative_example_bool)
                negative_example_entities_list.append(negative_example_entities)
                negative_example_entities_count_list.append(negative_example_entities_count)
        else:
            continue
    
    df = pd.DataFrame({
        'huric_id': huric_ids,
        'command': commands,
        'prompt': prompts,
        'score': scores,
        'bboxes': bboxes,
        'n_bboxes': [len(bbox) for bbox in bboxes],
        'labeled_image_path': labeled_images_paths,
        'labeled_bboxes': labeled_bboxes_list,
        'image_path': images_paths,
        'config_file': config_files,
        'prompt_type': prompt_types,
        'important_entities_to_include': important_entities_to_include_list,
        'important_entities_to_exclude': important_entities_to_exclude_list,
        'optional_entities': optional_entities_list,
        'frame_semantics': frame_semantics_list,
        'spatial_relations': spatial_relations_list,
        'excluded_entities_found': excluded_entities_found_indicators,
        'important_entity_missing': important_entity_not_found_indicators,
        'shot_type': shot_types,
        'negative_example_bool': negative_example_bool_list,
        'negative_example_entities': negative_example_entities_list,
        'negative_example_entities_count': negative_example_entities_count_list
    })

    df = df.sort_values(by=['huric_id', 'prompt_type', 'score'], ascending=[True, True, False]).reset_index(drop=True)

    df.to_csv(f'dataset_{TODAY}.csv', index=False)
    
    
def main():
    images_paths = []
    labeled_images_paths = []
    scores = []
    bboxes = []
    config_files = []
    commands = []
    huric_ids = []
    prompts = []
    prompt_types = []
    important_entities_to_include_list = []
    important_entities_to_exclude_list = []
    optional_entities_list = []
    frame_semantics_list = []
    spatial_relations_list = []
    excluded_entities_found_indicators = []
    important_entity_not_found_indicators = []
    shot_types = []
    negative_example_bool_list = []
    negative_example_entities_list = []
    negative_example_entities_count_list = []
    labeled_bboxes_list = []


    for image_path in tqdm(glob(f'{output_label_folder_path}**/*_original.png', recursive=True)):
        image_file_name = os.path.basename(image_path)
        current_folder = os.path.dirname(image_path)
        annotated_image_path = f"{current_folder}/{image_file_name.replace('original', 'annotated')}"

        if os.path.isfile(annotated_image_path):
            print(f"Skipping '{image_file_name}' as it is already processed. Check if it's present in the 'dataset.csv' file.")
            continue

        config_path = glob(f'{current_folder}/*_config.json')[0]
        with open(config_path, "r", encoding="utf8") as json_data:
            config = json.load(json_data)

        pil_image = Image.open(f'{image_path}')

        with open(f"{current_folder}/prompt.json", "r", encoding="utf8") as prompt_file:
            prompt_file_content = json.load(prompt_file)
            prompt = prompt_file_content['prompt']
            shot_type = prompt_file_content["shot_type"]

        huric_id = image_file_name.split('_')[0]
        command = config['command']
        entities_to_include = config['important_entities_to_include']
        entities_to_include_names = [e['type'] for e in entities_to_include]
        entities_to_include_ids = [e['atom'] for e in entities_to_include]
        entities_to_exclude = config['important_entities_to_exclude'] 
        entities_to_exclude_names = [e['type'] for e in entities_to_exclude] + ['person', 'robot']
        entities_to_exclude_ids = [e['atom'] for e in entities_to_exclude] + ['person_123456789', 'robot_123456789']
        optional_entities = config['optional_entities']
        frame_semantics = config['frame_semantics']
        spatial_relations = config['spatial_relations']
        prompt_type = config['prompt_type']
        negative_example_bool = config['negative_example'] 
        negative_example_entities = config['negative_example_entities'] 
        negative_example_entities_count = config['negative_example_entities_count'] 
        
        important_entities_to_include_list.append(entities_to_include)
        important_entities_to_exclude_list.append(entities_to_exclude)
        optional_entities_list.append(optional_entities)
        frame_semantics_list.append(frame_semantics)
        spatial_relations_list.append(spatial_relations)

        entity_names = entities_to_include_names + entities_to_exclude_names
        entity_ids = entities_to_include_ids + entities_to_exclude_ids

        image_source, image_preprocessed = preprocess_image(pil_image)
        annotated_frame, detected_bboxes, phrases, logits = detect(image_source, 
                                                image_preprocessed,
                                                text_prompt=" ".join(["a " + e.lower() + '.' for e in entity_names]), 
                                                model=groundingdino_model,
                                                box_threshold = 0.4,
                                                text_threshold = 0.3)
        
        annotations = list(zip(detected_bboxes, phrases, logits))
        json_annotations = list(zip(detected_bboxes.numpy().tolist(), phrases, entity_ids, logits.numpy().tolist()))

        with open(f'{current_folder}/annotations.txt', 'w') as f:
            json.dump(json_annotations, f)

        Image.fromarray(annotated_frame).save(annotated_image_path)

        score, important_entities_missing, entities_to_be_excluded_found, labeled_bboxes = scoring_function(entities_to_include, entities_to_exclude, annotations)

        scores.append(score)
        labeled_images_paths.append(annotated_image_path)
        bboxes.append(detected_bboxes.numpy().tolist())
        images_paths.append(image_path)
        config_files.append(config_path)
        commands.append(command)
        huric_ids.append(huric_id)
        prompts.append(prompt)
        prompt_types.append(prompt_type)
        important_entity_not_found_indicators.append(important_entities_missing)
        excluded_entities_found_indicators.append(entities_to_be_excluded_found)
        shot_types.append(shot_type)
        labeled_bboxes_list.append(labeled_bboxes)
        negative_example_bool_list.append(negative_example_bool)
        negative_example_entities_list.append(negative_example_entities)
        negative_example_entities_count_list.append(negative_example_entities_count)

    df = pd.DataFrame({
        'huric_id': huric_ids,
        'command': commands,
        'prompt': prompts,
        'score': scores,
        'bboxes': bboxes,
        'n_bboxes': [len(bbox) for bbox in bboxes],
        'labeled_image_path': labeled_images_paths,
        'labeled_bboxes': labeled_bboxes_list,
        'image_path': images_paths,
        'config_file': config_files,
        'prompt_type': prompt_types,
        'important_entities_to_include': important_entities_to_include_list,
        'important_entities_to_exclude': important_entities_to_exclude_list,
        'optional_entities': optional_entities_list,
        'frame_semantics': frame_semantics_list,
        'spatial_relations': spatial_relations_list,
        'excluded_entities_found': excluded_entities_found_indicators,
        'important_entity_missing': important_entity_not_found_indicators,
        'shot_type': shot_types,
        'negative_example_bool': negative_example_bool_list,
        'negative_example_entities': negative_example_entities_list,
        'negative_example_entities_count': negative_example_entities_count_list
    })

    df = df.sort_values(by=['huric_id', 'prompt_type', 'score'], ascending=[True, True, False]).reset_index(drop=True)

    df.to_csv('dataset.csv', index=False)


if __name__ == '__main__':
    main()
