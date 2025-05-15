import logging
import ast

logger = logging.getLogger(__name__)


def get_json_interpretation(interpretation):
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
                    frame_element_bb = ast.literal_eval(", ".join(frame_element.split("(")[1].split(",")[1:]).strip().replace(")", ""))
            except IndexError:
                logger.warning(f"IndexError: {frame_element}")
                raise
            
            frame_element_obj = {
                "name": frame_element_name,
                "surface": frame_element_value,
                "bbox_2d": frame_element_bb
            }
            
            frame_obj["elements"].append(frame_element_obj)
            
        json_interpretation.append(frame_obj)
        
    return json_interpretation
    

def export_tsv_json_dataset(df, output_file_path):
    column_headers = ["id", "image_path", "input", "output"]
    
    with open(output_file_path, "w", newline="") as csvfile:
        csvfile.write("\t".join(column_headers) + "\n")
        
        for idx, row in df.iterrows():
            id = row["huric_id"]
            image_path = row["image_path"]
            input = row["command"]
            output = get_json_interpretation(row["frame_semantics_bb"])
            csvfile.write(f"{id}\t{image_path}\t{input}\t{output}\n")
            
    logger.info(f"TSV file with json saved to {output_file_path}")