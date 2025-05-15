import os
import ast
import logging
import re
from filter_top_k_images_and_export.huric.utils import convert_box_format_from_xywd_to_xyxy, convert_box_format_from_relative_xyxy_to_pixel_xyxy_4minicpm, get_bounding_box_tag

logger = logging.getLogger(__name__)


def parse_frame_element(frame_element):
    """
    It parses the frame element and removes unnecessary characters.
    It also replaces some words with their singular form.
    """
    
    frame_element = frame_element.replace("(", "").replace(")", "").replace(",", "").strip()
    
    frame_element = re.sub(r"\b(?:a|to|the|near|next|on)\b", "", frame_element)
    frame_element = re.sub(r"\b(?:books)\b", "book", frame_element)
    
    return frame_element.strip()


def get_room_name_splitted(lexical_refs):
    """
    It loops through the lexical references and checks if any of them is a room name.
    If it is, it returns the room in a splitted way: livingroom -> living room.
    """
    
    room_names = [
        "livingroom", "bedroom", "bathroom", "diningroom", "kitchen", "hallway", "garage", "garden", "office", "laundryroom"
    ]
    
    for ref in lexical_refs:
        try:
            index = room_names.index(ref.replace(" ", ""))
            return room_names[index]
        except ValueError:
            continue
        
    return None


def parse_and_get_lexical_references(input_lexical_refs):
    """
    It parses the lexical references and returns them in a list.
    It also adds the room names in a splitted way.
    """
    
    lexical_refs = [" ".join(ref.split("_")).lower() for ref in input_lexical_refs]
    lexical_refs += ["".join(ref.lower().split(" ")) for ref in input_lexical_refs]
    room_names = get_room_name_splitted(lexical_refs) 
    if room_names != None: lexical_refs += [room_names]
    return list(set(lexical_refs))


def is_only_one_room_element(entity_reference_map):
    """
    It loops through the entity reference map and checks if there is only one room element.
    If there is, it returns the room element.
    """
    
    room_elements = []
    
    for entity_id, entity_info in entity_reference_map.items():
        for entity_lr in parse_and_get_lexical_references(entity_info.get("lexical_references", [])):
            if "room" in entity_lr:
                room_elements.append(entity_id)
    
    return True if len(room_elements) == 1 else False


def add_bb_to_interpretation(interpretation, image_path, entity_reference_map, verbose: bool = False):
    """
    It adds the bounding boxes to the interpretation.
    It replaces the <ADD_HERE_BOUNDING_BOX> tag with the bounding box coordinates.
    """
    
    bb_file_path = image_path.replace(image_path.split("/")[-1], "annotations.txt")
    if not os.path.exists(bb_file_path):
        raise FileNotFoundError(f"Bounding box file not found: {bb_file_path}")

    with open(bb_file_path, 'r') as file:
        file_data = file.read().strip()
        bb_data = ast.literal_eval(file_data)
    
    if verbose:
        logger.warning(f"Loaded {len(bb_data)} bounding boxes from {bb_file_path}")

    for bb in bb_data:
        bb_coordinates = bb[0]
        bb_label = bb[1].lower().replace("a ", "").replace("_", " ").strip()
        
        if verbose:
            logger.warning(f"\nEvaluating BB label: '{bb_label}'")

        for frame_string in interpretation.split(" - "):
            frame_elements = "(".join(frame_string.split("(")[1:]).split("),")
            for frame_element in frame_elements:
                if "(" not in frame_element:
                    continue
                try:
                    frame_element_value = frame_element.split("(")[1].split(",")[0].strip().lower()
                    frame_element_bb_tag = frame_element.split("(")[1].split(",")[1].strip()
                except IndexError:
                    continue
                
                if verbose:
                    logger.warning(f"\t→ Frame Element: '{frame_element_value}' with tag: {frame_element_bb_tag}")
                
                if "<ADD_HERE_BOUNDING_BOX>" in frame_element_bb_tag:
                    found_bb_element = False
                    for entity_id, entity_info in entity_reference_map.items():
                        parsed_frame_element = parse_frame_element(frame_element_value)
                        
                        lexical_refs = parse_and_get_lexical_references(entity_info.get("lexical_references", []))

                        label_match = bb_label in lexical_refs
                        value_match = parsed_frame_element in lexical_refs

                        if verbose:
                            logger.warning(f"\t\tChecking entity lexical_refs: {lexical_refs}")
                            logger.warning(f"\t\t→ label_match: {label_match}, value_match: {value_match}")

                        if (
                            bb_label in lexical_refs and
                            parsed_frame_element in lexical_refs
                        ):
                            boxes_xyxy = convert_box_format_from_xywd_to_xyxy(bb_coordinates)
                            boxes_xyxy = convert_box_format_from_relative_xyxy_to_pixel_xyxy_4minicpm(boxes_xyxy)
                            new_frame_element = frame_element.replace(
                                '<ADD_HERE_BOUNDING_BOX>',
                                f'[{boxes_xyxy[0]}, {boxes_xyxy[1]}, {boxes_xyxy[2]}, {boxes_xyxy[3]}]'
                            )
                            
                            if verbose:
                                logger.warning(f"\t\t\t✔ Replacing BB for: {frame_element_value} ← from entity '{bb_label}' with BB: {boxes_xyxy}")
                            
                            interpretation = interpretation.replace(frame_element, new_frame_element)
                            found_bb_element = True
                            break
                    
                    # if the label is "room" and there is only one room element, replace the BB
                    if not found_bb_element and bb_label == "room" and is_only_one_room_element(entity_reference_map) != None:
                        boxes_xyxy = convert_box_format_from_xywd_to_xyxy(bb_coordinates)
                        boxes_xyxy = convert_box_format_from_relative_xyxy_to_pixel_xyxy_4minicpm(boxes_xyxy)
                        new_frame_element = frame_element.replace(
                            '<ADD_HERE_BOUNDING_BOX>',
                            f'[{boxes_xyxy[0]}, {boxes_xyxy[1]}, {boxes_xyxy[2]}, {boxes_xyxy[3]}]'
                        )
                        
                        if verbose:
                            logger.warning(f"\t\t\t✔[only1room] Replacing BB for: {frame_element_value} ← from entity '{bb_label}' with BB: {boxes_xyxy}")
                        
                        interpretation = interpretation.replace(frame_element, new_frame_element)
                        found_bb_element = True
                            
                            

    if verbose:
        logger.warning(f"\n{interpretation.replace('<ADD_HERE_BOUNDING_BOX>', '<MISSING>')}")

    return interpretation.replace("<ADD_HERE_BOUNDING_BOX>", "<MISSING>")



def compute_frame_semantics_with_bb(row, entity_reference_map, subset_list: list = []):
    """
    It computes the frame semantics with bounding boxes.
    It loops through the frame semantics and adds the bounding boxes to the interpretation.
    """
    
    result = ""

    for frame_string in row["frame_semantics"]:
        frame_name = frame_string.split("Frame: ")[1].split(" (")[0].replace('"', "").replace("'", "").strip().upper()

        if result == "":
            result += f"{frame_name}("
        else:
            if result.endswith(", "):
                result = result[:-2]
            if not result.endswith("))"):
                result += ")"
            result += f" - {frame_name}("

        frame_elements = frame_string.split("• ")
        for frame_element in frame_elements:
            if "Frame Element" in frame_element:
                frame_element_name = frame_element.split("Frame Element: ")[1].split("→")[0].replace('"', "").replace("'", "").strip()
                frame_element_value = frame_element.split("→")[1].split("(")[0].replace('"', "").replace("'", "").strip().lower()
                bb_value = get_bounding_box_tag(frame_element_value)
                result += f"{frame_element_name}({frame_element_value}, {bb_value}), "

    result = result.rstrip(", ") + ")"
    
    # sets verbose only if the huric_id is in the subset_list
    # this is done to avoid printing the entire dataframe
    verbose = int(row["huric_id"].replace("_enriched", "").replace(".hrc", "")) in subset_list
    
    result = add_bb_to_interpretation(result, row["image_path"], entity_reference_map, verbose=False)
    
    return result
