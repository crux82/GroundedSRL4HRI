import json
import os

# Input path
data_name="json_top_1_images"
input_file = '../image_generator/data/dataset/'+data_name+'.tsv'
directory_image="../image_generator/data/images/"
output_file='./dataset/top1/complete/'+data_name+'.json'



def convert_line_to_json(line):
    parts = line.split('\t')
    id_part = parts[0].strip()
    filename = os.path.basename(parts[1].strip())
    path = directory_image+filename

    request = parts[2].strip()
    text = parts[3].strip()
    prompt='''You are given a natural language command and a corresponding image depicting a domestic environment. Your task is to convert the command into a **structured, grounded semantic representation** in the form of **predicate-argument structures**.

This means you must:
- Identify the semantic **frames** expressed in the command (e.g., MOTION, TAKING, BRINGING, LOCATING, PLACING, CHANGE_OPERATIONAL_STATE, etc.)
- For each frame, extract a list of **frame elements**, grounded visually whenever possible.

Your output must be a **list of dictionaries**, one per frame. Each dictionary contains:
- "frame": the name of the frame (string), following FrameNet conventions.
- "elements": a list of dictionaries, each with:
  - "name": the role of the frame element (e.g., Theme, Goal, Agent, Beneficiary, etc.)
  - "surface": the exact text span from the command. It must be the semantic head, i.e., the word referring to the object.
  - "bbox_2d": either
    - a list of four integers [x1, y1, x2, y2] representing a visible object's bounding box in the image
    - or a symbolic tag among <ROBOT>, <PERSON>, <ROOM>, <POSITION>, <STATUS>, <MISSING> and <ITEM>

### Important guidelines:
- Only annotate what is **visually present** in the image and it is relevant for the command (as a frame element).
- Use <MISSING> if the referred object **cannot be seen**.
- Use <ROBOT> for the robot, and <PERSON> for a human addressee.
- Use <POSITION> when the target is an abstract spatial reference like "there" or "here".
- Use <STATUS> for expressions of operational state (e.g., "on", "off").
- Use <ITEM> for references to unspecified or deictic objects (e.g., pronouns like "this", "that", or "it" when referring to a visible item).
- All tags must be typed **exactly as given**, with no added characters or typos.
- Always extract **surface** spans literally from the input command.
- Your output must be in **valid JSON**, and formatted as **a single line**.

---

### Examples:

**Command:**  
<photo>  
could you go to the kitchen please

**Output:**  
[{'frame': 'MOTION', 'elements': [{'name': 'Agent', 'surface': 'you', 'bbox_2d': '<ROBOT>'}, {'name': 'Goal', 'surface': 'kitchen', 'bbox_2d': '<ROOM>'}]}]

---

**Command:**  
<photo>  
bring me the bottle

**Output:**  
[{'frame': 'BRINGING', 'elements': [{'name': 'Agent', 'surface': 'you', 'bbox_2d': '<ROBOT>'}, {'name': 'Theme', 'surface': 'bottle', 'bbox_2d': [210, 612, 780, 830]}, {'name': 'Beneficiary', 'surface': 'me', 'bbox_2d': '<PERSON>'}]}]

---

**Command:**  
<photo>  
take the wallet from the table

**Output:**  
[{'frame': 'TAKING', 'elements': [{'name': 'Agent', 'surface': 'you', 'bbox_2d': '<ROBOT>'}, {'name': 'Theme', 'surface': 'wallet', 'bbox_2d': '<MISSING>'}, {'name': 'Source', 'surface': 'table', 'bbox_2d': [480, 660, 960, 900]}]}]

---

**Command:**  
<photo>  
turn on the tv

**Output:**  
[{'frame': 'CHANGE_OPERATIONAL_STATE', 'elements': [{'name': 'Agent', 'surface': 'you', 'bbox_2d': '<ROBOT>'}, {'name': 'Operational_state', 'surface': 'on', 'bbox_2d': '<STATUS>'}, {'name': 'Device', 'surface': 'tv', 'bbox_2d': [251, 337, 832, 708]}]}]

---

Now do the same for the following example:'''
    return {
        "id": id_part,
        "image": path,
        "conversations": [
            {
                "role": "user",
                "content": prompt+"\n<image>\n"+request
            },
            {
                "role": "assistant",
                "content": text
            }
        ]
    }

def convert_file(input_file):
    output_data = []
    
    with open(input_file, 'r') as file:
        next(file)  

        for line in file:
            print(line)
            json_entry = convert_line_to_json(line)
            output_data.append(json_entry)
    
    return output_data

output_data = convert_file(input_file)

with open(output_file, 'w') as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)

print("END")
