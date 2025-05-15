import xml.etree.ElementTree as ET

def extract_frame_semantics(huric_file_path, root, tokens):
    frame_semantics = []
    for item in root.find("commands").find("command").find("semantics").find("frames").findall("frame"):
        frame_name = item.attrib['name']
        lex_unit = item.find("lexicalUnit").find("token").attrib['id']
        lemma_frame_name = tokens[int(lex_unit) - 1]
        frame_elements = item.find("frameElements").findall("frameElement")
        frame_phrase = f"Frame: \"{frame_name}\" (evoked in the command by \"{lemma_frame_name}\")\n\t• "
        frame_element_phrases = []
        for frame_element in frame_elements:
            frame_element_type = frame_element.attrib['type']
            if 'semanticHead' not in frame_element.attrib:
                frame_element_semantic_head = frame_element.find("token").attrib['id']
            else:
                frame_element_semantic_head = frame_element.attrib['semanticHead']
            if int(frame_element_semantic_head) > len(tokens):
                print("Warning: Semantic head", frame_element_semantic_head, "is out of bounds for tokens list in huric file", huric_file_path, "so the id is ignored and will be taken the last token id declared inside the Frame Element.")
                frame_element_semantic_head = frame_element.findall("token")[-1].attrib['id']
            frame_element_semantic_head_word = tokens[int(frame_element_semantic_head)-1]
            frame_element_surfaces = []
            for t in frame_element.findall("token"):
                if int(t.attrib['id']) <= len(tokens):
                    frame_element_surfaces.append(tokens[int(t.attrib['id'])-1])
                else:
                    print("Warning: Token not found in tokens list:", t.attrib['id'], "for huric file", huric_file_path, "so the id is ignored.")
            
            # if the semantic head is "room" and the previous token is "living", "bed" or "bath", merge them
            if frame_element_semantic_head_word == "room":
                try:
                    if frame_element_surfaces[frame_element_surfaces.index(frame_element_semantic_head_word)-1] in ["living", "bed", "bath"]:
                        frame_element_semantic_head_word = f"{frame_element_surfaces[frame_element_surfaces.index(frame_element_semantic_head_word)-1]} {frame_element_semantic_head_word}"
                except Exception as e:
                    print("Error merging semantic head with previous token:", e, "\n\n")
                    quit(frame_element_surfaces)
                
            frame_element_phrases.append(f"Frame Element: \"{frame_element_type}\" → \"{frame_element_semantic_head_word.capitalize()}\" (represented in the command by \"{' '.join(frame_element_surfaces)}\").")
        frame_phrase += "\n\t• ".join(frame_element_phrases)
        frame_semantics.append(frame_phrase)
    return frame_semantics


def parse_huric_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    command_element = root.find('commands').find('command')
    command = command_element.find('sentence').text
    tokens = [token.attrib['surface'] for token in command_element.find('tokens').findall('token')]

    frame_semantics = extract_frame_semantics(file_path, root, tokens)

    entities = {
        "visible": [], "missing": [], "status": [],
        "near_to": [], "far_from": [], "ontop": [], "inside": []
    }

    entity_reference_map = {}

    for entity in root.findall(".//entity"):
        entity_name = entity.get("atom", "")

        for attr in entity.findall(".//attribute"):
            attr_name = attr.get("name", "")
            if attr_name in entities:
                if attr_name == "visible":
                    if attr.find("value") is not None:
                        if attr.find("value").text.strip() == "true":
                            entities[attr_name].append(entity_name)
                        else:
                            entities["missing"].append(entity_name)
                    else:
                        entities["missing"].append(entity_name)
            elif attr_name == "lexical_references":
                # lexical_refs = [ref.text.strip() for ref in entity.findall(".//lexicalReference") if ref.text]
                lexical_refs = [val.text.strip() for val in attr.findall("value") if val.text]
                entity_reference_map[entity_name] = {
                    "atom": entity_name,
                    "lexical_references": lexical_refs
                }

    for entity in root.findall(".//entity"):
        entity_name = entity.get("atom", "")
        for attr in entity.findall(".//attribute"):
            attr_name = attr.get("name", "")
            if attr_name in entities:
                if attr_name == "status" and entity_name in entities["visible"]:
                    if attr.find("value") is not None:
                        status = attr.find("value").text.strip()
                        entities[attr_name].append(f"{status} - {entity_name}")
                else:
                    related_entity = attr.find(".//value/entity")
                    if related_entity is not None and entity_name in entities["visible"]:
                        related_name = related_entity.get("atom", "")
                        entities[attr_name].append(f"{entity_name} - {related_name}")

    return command, frame_semantics, entities, entity_reference_map
