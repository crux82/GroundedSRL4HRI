import os
import xml.etree.ElementTree as ET
import json
import random
import math
from ast import literal_eval
from datetime import datetime

# MiniCPM setup
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch

TODAY = datetime.now().strftime("%Y%m%d")

# Directory paths
HURIC_DIR = "path/to/huric/files" # Replace with the actual path to huric_dir
if HURIC_DIR.equals("path/to/huric/files"):
    raise ValueError("Please set the correct path to huric_dir in line 17!.")
IMAGES_DIR = "./data/output_label_folder"
OUTPUT_FILE = f"./data/consistency_check_scores_{TODAY}.json"
TOTAL_IMAGES = 39595 # for logging purposes, update this if it's inconsistent


# Load the Visual LLM
model_id = "openbmb/MiniCPM-V-2_6"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    attn_implementation='sdpa',
    torch_dtype=torch.float16, 
    device_map="auto"
)


# Properties of interest
PROPERTIES = {"visible", "status", "near_to", "far_from", "ontop", "inside"}

def parse_huric_file(file_path):
    """Extract relevant information from a Huric XML file."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    command_element = root.find('commands').find('command')
    command = command_element.find('sentence').text.strip() if command_element is not None else None
    entity_data = {}

    human_entity, robot_entity = False, False
    
    for entity in root.findall(".//entity"):
        entity_id = entity.get("atom")
        entity_type = entity.get("type")
        human_entity = True if "person" in entity_type.lower() or "me" in entity_type.lower() else human_entity
        robot_entity = True if "robot" in entity_type.lower() or "you" in entity_type.lower() else robot_entity
        attributes = {}
        
        for attr in entity.findall(".//attribute"):
            attr_name = attr.get("name")
            if attr_name in PROPERTIES:
                if attr_name in {"visible", "status"}:  # Direct values
                    value = attr.find("value").text.strip() if attr.find("value") is not None else None
                else:  # Entity-based attributes
                    value_element = attr.find("value/entity")
                    value = (value_element.get("type"), value_element.get("atom")) if value_element is not None else None
                
                if value is not None:
                    attributes[attr_name] = value
        
        if attributes:
            entity_data[entity_id] = {"type": entity_type, "attributes": attributes}

    # create "person" and "robot" mock entities if not present
    # we don't want to have them in the images, so we need them for the questions
    if not human_entity:
        entity_data["person_0123456789"] = {
            "type": "person",
            "attributes": {
                "visible": False,
            }
        }
    
    if not robot_entity:
        entity_data["robot_0123456789"] = {
            "type": "robot",
            "attributes": {
                "visible": False,
            }
        }
    
    return command, entity_data


def generate_questions(entity_data):
    """Generate questions for the Visual LLM."""
    questions = []
    
    for entity_id, data in entity_data.items():
        entity_type = data["type"]
        for attr, expected_value in data["attributes"].items():
            related_entity = {}
            if attr == "visible":
                # This was the original question
                # question = f"Is the '{entity_type}' ({entity_id}) entity visible? Answer only yes or no."
                # NOW we check if the entity is visible using Grounding DINO
                # it will be reverted back to the original question after we get the answer
                question = "ASK GROUNDING DINO"
                expected_value = "yes" if expected_value and expected_value == "true" else "no"
                status = "visible" if expected_value and expected_value == "yes" else "missing"
            elif attr == "status":
                question = f"Is the '{entity_type}' ({entity_id}) entity {expected_value}? Answer only yes or no."
                status = expected_value
                expected_value = "yes"  # Assume the status is true
            else:
                related_entity_type, related_entity_id = expected_value
                if attr == "ontop":
                    question = f"Is the '{entity_type}' ({entity_id}) entity on top of the '{related_entity_type}' ({related_entity_id})? Answer only yes or no."
                    status = "on top"
                else:
                    question = f"Is the '{entity_type}' ({entity_id}) entity {attr.replace('_', ' ')} the '{related_entity_type}' ({related_entity_id})? Answer only yes or no."
                    status = attr.replace("_", " ")

                expected_value = "yes"  # Assume the relationship is true
                related_entity = {"entity_id": related_entity_id, "entity_type": related_entity_type}
            
            questions.append({
                "entity_id": entity_id,
                "entity_type": entity_type,
                "attribute": attr,
                "status": status,
                "related_entity": related_entity,
                "expected": expected_value,
                "question": question
            })
    
    return questions


def get_answer_and_confidence(prediction, expected_answer):
    """Extract the answer and confidence from the LLM's prediction."""
    output_text = prediction.strip().lower()
    if output_text.startswith("yes"):
        answer = "yes"
        conf = (0.9, 0.99) if "yes" in expected_answer else (0.1, 0.01)
    elif output_text.startswith("no"):
        answer = "no"
        conf = (0.9, 0.99) if "no" in expected_answer else (0.1, 0.01)
    elif " yes " in output_text and " no " in output_text:
        answer = "yes"
        conf = (0.5, 0.5)
    else:
        answer = "no"
        conf = (0.5, 0.5)
    
    return answer, conf


def get_answer_with_grounding_dino(annotation_file, entity_type, expected_answer):
    """Get the answer using Grounding DINO."""
    ann = []
    with open(annotation_file, 'r') as f:
        data = f.read()
        ann = literal_eval(data)

    for el in ann:
        print(el)
        _, type, entity_id, conf = el
        if entity_type.lower() in type.lower():
            if expected_answer == "yes":
                return "yes", (conf, conf)
            else:
                return "yes", (1 - conf, 1 - conf)
    
    # since MiniCPM does not produce any confidence for its answers, we simulate it
    if expected_answer == "yes":
        return "no", (0.1, 0.01)
    else:
        return "no", (0.9, 0.99)


def answer_with_minicpm(image_path, questions, annotation_files):
    image = Image.open(image_path).convert("RGB")
    results = []

    for q, ann_file in zip(questions, annotation_files):

        if q['question'] == "ASK GROUNDING DINO":
            answer, confidences = get_answer_with_grounding_dino(ann_file, q['entity_type'], q["expected"])
            # revert back the question to natural language after we used Grounding DINO
            question = f"Is the '{q['entity_type']}' ({q['entity_id']}) entity visible? Answer only yes or no."
            q['question'] = question

        else:
            msgs = [{"role": "user", "content": [image, q['question']]}]

            # this method does not produce any confidence for its answers
            # so we simulate it later
            output = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling = False,  
                system_prompt = "You are a robot that can see and understand images. You are asked to answer questions about the image.",
                max_new_tokens=10,
            )
            
            answer, confidences = get_answer_and_confidence(output, q["expected"])

        result = q.copy()
        result.update({
            "answer": answer,
            "confidences": confidences
        })
        results.append(result)

    return results


@DeprecationWarning
def generate_simulated_responses(questions):
    """Generate realistic responses based on the question type."""
    responses = []
    
    for q in questions:
        attr = q["attribute"]
        
        if attr == "visible" or attr in {"near_to", "far_from", "ontop", "inside"}:
            answer = random.choice(["yes", "no"])
            confidence = random.uniform(0.5, 1.0)
        elif attr == "status":
            answer = random.choice(["on", "off", "closed", "open"])
            confidence = random.uniform(0.5, 1.0)
        
        responses.append({"answer": answer, "confidence": confidence})
    
    return responses


@DeprecationWarning
def evaluate_responses(questions, responses):
    """Compute the log-score based on the LLM's responses to avoid numerical underflow."""
    log_score = 0.0
    results = []
    
    for q, r in zip(questions, responses):
        confidence = r["confidence"]
        answer = r["answer"].strip().lower()
        expected = q["expected"].strip().lower()
        
        prob = confidence if answer == expected else 1 - confidence
        
        log_score += math.log(prob + 1e-9)  # Avoid log(0)
        q["answer"] = answer
        q["confidence"] = confidence
        results.append(q)
    
    final_score = math.exp(log_score)  # Convert back from log-space
    return final_score, results


# Example response structure:
# responses = [
#   {
#     "entity_id": entity_id,
#     "entity_type": entity_type,
#     "attribute": attr,
#     "status": status,
#     "related_entity": related_entity,
#     "expected": expected_value,
#     "question": question,
#     "answer": answer,
#     "confidences": confidences
#   }, {...}
# ]
def compute_scores(responses):
    scores = []
    should_be_visible = {}
    for response in responses:
        if response['attribute'] == "visible" and response['expected'] == "yes":
            should_be_visible[response['entity_id']] = True
    
    for response in responses:
        if response['attribute'] != "visible":
            if should_be_visible.get(response['entity_id'], False):
                scores.append(response['confidences'])
        else:
            scores.append(response['confidences'])

    score_09 = math.exp(sum([math.log(score[0]) for score in scores]))
    score_099 = math.exp(sum([math.log(score[1]) for score in scores]))

    return score_09, score_099


def process_all_huric_files():
    """Process all Huric files and compute scores for corresponding images."""
    all_results = {}
    
    count = 0
    for huric_root, _, huric_files in os.walk(HURIC_DIR):
        for filename in huric_files:
            if filename.endswith("_enriched.hrc"):
                file_path = os.path.join(huric_root, filename)
                print(50*"*")
                print(f"Processing {file_path}...\n")
                
                command, entity_data = parse_huric_file(file_path)
                questions = generate_questions(entity_data)
                
                image_folder = os.path.join(IMAGES_DIR, filename.replace("_enriched.hrc", ""))
                image_files, annotation_files = [], []
                for images_root, _, images_file in os.walk(image_folder):
                    for f in images_file:
                        if f.lower().endswith((".jpg", ".png")) and "original" in f.lower():
                            image_files.append(os.path.join(images_root, f))
                        if f.lower().endswith(".txt") and "annotation" in f.lower():
                            annotation_files.append(os.path.join(images_root, f))
                if not image_files:
                    print(f"No image found for {filename}")
                    continue
                if not annotation_files:
                    print(f"No annotation found for {filename}")
                    continue

                all_image_responses = []
                for image_file_path in image_files:
                    count += 1
                    print(f"{count}/{TOTAL_IMAGES}: Processing image {image_file_path}\n")
                    
                    image_path = os.path.join(image_folder, image_file_path)
                    responses = answer_with_minicpm(image_path, questions, annotation_files)
                    score_09, score_099 = compute_scores(responses)
                    all_image_responses.append({"image": image_file_path, "responses": responses, "scores": (score_09, score_099)})

                    print(f"Image: {image_file_path}", "\n", f"Scores: ({score_09}, {score_099})\n")
                    

                all_results[filename] = {"command": command, "images": all_image_responses}
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(50*"*")
    print(50*"*")
    print(f"Processing complete. Results saved in '{OUTPUT_FILE}'.")
    print(50*"*")
    print(50*"*")
    


def process_huric_ids(huric_ids):
    """Process all Huric files and compute scores for corresponding images."""
    all_results = {}
    
    count = 0
    for huric_root, _, huric_files in os.walk(HURIC_DIR):
        for filename in huric_files:
            file_path = os.path.join(huric_root, filename)
            if filename.endswith("_enriched.hrc") and filename.replace("_enriched.hrc", "") in huric_ids:
                print(50*"*")
                print(f"Processing {file_path}...\n")
                
                command, entity_data = parse_huric_file(file_path)
                questions = generate_questions(entity_data)
                
                image_folder = os.path.join(IMAGES_DIR, filename.replace("_enriched.hrc", "") + "_" + TODAY)
                if not os.path.exists(image_folder):
                    print(f"Image folder '{image_folder}' does not exist. Skipping...")
                    continue
                image_files, annotation_files = [], []
                for images_root, _, images_file in os.walk(image_folder):
                    for f in images_file:
                        if f.lower().endswith((".jpg", ".png")) and "original" in f.lower():
                            image_files.append(os.path.join(images_root, f))
                        if f.lower().endswith(".txt") and "annotation" in f.lower():
                            annotation_files.append(os.path.join(images_root, f))
                if not image_files:
                    print(f"No image found for {filename}")
                    continue
                if not annotation_files:
                    print(f"No annotation found for {filename}")
                    continue

                all_image_responses = []
                for image_file_path in image_files:
                    count += 1
                    print(f"{count}/{TOTAL_IMAGES}: Processing image {image_file_path}\n")
                    
                    image_path = os.path.join(image_folder, image_file_path)
                    responses = answer_with_minicpm(image_path, questions, annotation_files)
                    score_09, score_099 = compute_scores(responses)
                    all_image_responses.append({"image": image_file_path, "responses": responses, "scores": (score_09, score_099)})

                    print(f"Image: {image_file_path}", "\n", f"Scores: ({score_09}, {score_099})\n")
                    

                all_results[filename] = {"command": command, "images": all_image_responses}
    
    # OUTPUT_FILE is a path and already contains TODAY
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=4)
        

if __name__ == "__main__":
    process_all_huric_files()
