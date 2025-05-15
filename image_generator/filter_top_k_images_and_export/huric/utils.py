import os

def get_hrc_file_path(huric_dir, huric_id):
    for root, _, files in os.walk(huric_dir):
        for file in files:
            if "dummy_subset" in huric_dir:
                if file.endswith(".hrc") and huric_id.replace("_enriched.hrc", "") in file:
                    return os.path.join(root, file)
            else:
                if file.endswith(".hrc") and huric_id in file and "_enriched" in file:
                    return os.path.join(root, file)
    return ""


def convert_box_format_from_xywd_to_xyxy(bb):
    """
    Convert a bounding box from center format (x_center, y_center, width, height) to corner format (x1, y1, x2, y2).
    """
    x_center, y_center, width, height = bb
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)
    return x1, y1, x2, y2


def convert_box_format_from_relative_xyxy_to_pixel_xyxy(bb):
    """
    Convert a bounding box from a relative corner format (x1, y1, x2, y2) to a pixel corner format.
    Assumes the image is 1024x1024.
    """
    x1, y1, x2, y2 = bb
    x1 = int(x1 * 1024)
    y1 = int(y1 * 1024)
    x2 = int(x2 * 1024)
    y2 = int(y2 * 1024)
    
    return x1, y1, x2, y2


def convert_box_format_from_relative_xyxy_to_pixel_xyxy_4minicpm(bb):
    """
    Convert a bounding box from a relative corner format (x1, y1, x2, y2) to a pixel corner format.
    Assumes the image is 1024x1024.
    It adds 0s to the bounding box coordinates to make them 3 digits long.
    """
    x1, y1, x2, y2 = bb
    x1 = int(x1 * 1024)
    y1 = int(y1 * 1024)
    x2 = int(x2 * 1024)
    y2 = int(y2 * 1024)
    
    # x1 = str(x1).zfill(3)
    # y1 = str(y1).zfill(3)
    # x2 = str(x2).zfill(3)
    # y2 = str(y2).zfill(3)
    
    return x1, y1, x2, y2


def get_bounding_box_tag(frame_element_value):
    exceptions_list_robot = ["you", "robot"]

    exceptions_list_people = [
        "i", "they", "them", "he", "she", "we", "us", "me", "person", "people", "human",
        "mother", "father", "john", "sister", "daniel", "marco", "michael", "mark", "postman", "friend", 
        "man", "woman", "guy",
    ]

    exceptions_list_rooms = [
        "room", "kitchen", "living", "bathroom", "bedroom", "garage", "garden", "dining", "office",
        "laundry", "hallway", "livingroom", "diningroom", "bath room", "bed room", "kitchen room",
        "laundryroom", "living room", "dining room",
    ]

    exceptions_list_speed = [
        "fast", "slow",
    ]
    
    exceptions_list_items = [
        "this", "that", "these", "those", "it", "them", "one", "two", "three", 
        "four", "five", "item", "hand", "guest", "architecture",  "line",
    ]

    exceptions_list_position = [
        "here", "there", "front", "back", "right", "left", "up", "down",
        "high", "low", "near", "far", "close", "away", "inside", "outside",
    ]

    exceptions_list_status = [
        "start", "status", "pace", "end",
        "side", "lot", "bit", "set", "on", "off", "light",
    ]

    exceptions_list = (
        exceptions_list_people + exceptions_list_rooms + exceptions_list_speed + 
        exceptions_list_position + exceptions_list_status + exceptions_list_robot
    )
    
    if frame_element_value.lower() in exceptions_list:
        if frame_element_value.lower() in exceptions_list_robot:
            return f"<ROBOT>"
        if frame_element_value.lower() in exceptions_list_people:
            return f"<PERSON>"
        elif frame_element_value.lower() in exceptions_list_rooms:
            return f"<ROOM>"
        elif frame_element_value.lower() in exceptions_list_speed:
            return f"<SPEED>"
        elif frame_element_value.lower() in exceptions_list_items:
            return f"<ITEM>"
        elif frame_element_value.lower() in exceptions_list_position:
            return f"<POSITION>"
        elif frame_element_value.lower() in exceptions_list_status:
            return f"<STATUS>"
        else:
            return f"<OTHER>"
    
    return "<ADD_HERE_BOUNDING_BOX>"
