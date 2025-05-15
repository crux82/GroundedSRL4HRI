import argparse
import os
import xml.etree.ElementTree as ET
import math
import configparser
from glob import glob
import json
import itertools


def generate_metaprompts_for_huric_ids(huric_ids, types_list: list = None):
	# [Execution]
	prompt_template_folder = "/data/chromei/domestic_image_generator/prompt_generation/template_folder/"
	output_folder = "/data/chromei/domestic_image_generator/data/output_folder/"
	# huric_folder = "/data/chromei/huric_procthor_data/huric_procthor/en/"
	huric_folder = "/data/chromei/huric_procthor_data/huric_procthor/en_enriched/"

	# [Entities]
	entity_types_to_exclude = ["it", "robot", "me", "user", "person", "you"]
	rooms = ["bedroom", "kitchen", "livingroom", "bathroom", "corridor", "diningroom", "studio"]

	if len(huric_ids) == 0:
		raise Exception("No huric ids provided in config file. Provide specific ids separated by a comma to process only specific huric files or 'all' to process all huric files.")
	# print(huric_ids)

	process_all = False

	if 'all' in huric_ids:
		process_all = True
	
	huric_files = glob(f"{huric_folder}**/**.hrc")
	huric_files_to_process = []

	for huric_id in huric_ids:
		for f in huric_files:
			if (process_all or huric_id in f) and "_enriched" in f:
				huric_files_to_process.append(f.replace("\\", "/"))
	# print(huric_files_to_process)

	for huric_file in huric_files_to_process:
		if types_list is None or len(types_list) == 0:
			process_huric_file(huric_file, output_folder, prompt_template_folder, rooms, entity_types_to_exclude)
		else:
			for config_type in types_list:
				process_huric_file_by_type(huric_file, output_folder, prompt_template_folder, rooms, entity_types_to_exclude, config_type)

	return


def process_huric_file_by_type(huric_file_path, output_folder, prompt_template_folder, rooms, entity_types_to_exclude, configuration_type):
	important_entities = []
	optional_entities = []
	huric_rooms = []
	spatial_relations = []
	frame_semantics = []
	
	lexical_groundings = []

	tokens = []

	entity_subsets = []
	far_relation_dict = {}
	close_relation_dict = {}

	prompt_templates = get_prompt_templates(prompt_template_folder)
	os.makedirs(output_folder, exist_ok=True)


	# print(f"Analyzing Huric file {huric_file_path}")
	with open(huric_file_path,"r") as f:
		tree = ET.parse(f)
		root = tree.getroot()

		id_huric = root.attrib['id']

		# Extract important entities
		for item in root.find('lexicalGroundings').findall('lexicalGrounding'):
			atom = item.attrib['atom']
			token_id = item.attrib['tokenId']
			lexical_groundings.append({
				"atom": atom,
				"atom_id": atom.split("_")[1],
				"token_id": token_id
			})
		
		# Extract commands
		command_element = root.find('commands').find('command')
		command = command_element.find('sentence').text
		for token_element in command_element.find('tokens').findall('token'):
			tokens.append(token_element.attrib['lemma'])


		# Min and max dimensions to determine the ambient max dimensions
		min_x = 1000
		max_x = -1000
		min_y = 1000
		max_y = -1000
		# Extract optional entities
		for item in root.find("semanticMap").find("entities").findall("entity"):
			atom = item.attrib['atom']
			entity_type = item.attrib['type']
			atom_id = atom.split("_")[1]
			coord = item.find('coordinate')
			x = float(coord.attrib['x'])
			y = float(coord.attrib['y'])

			# Extract ambient max dimensions
			if x < min_x:
				min_x = x
			if x > max_x:
				max_x = x
			if y < min_y:
				min_y = y
			if y > max_y:
				max_y = y

			if entity_type.lower() in rooms:
				# print(f"Room found: {entity_type}")
				huric_room = {
					"room": entity_type.capitalize(),
					"x": x,
					"y": y
				}
				huric_rooms.append(huric_room)
			elif atom_id not in [lexical_grounding["atom_id"] for lexical_grounding in lexical_groundings]:
				if entity_type.lower() not in entity_types_to_exclude and entity_type.lower() not in rooms:
					optional_entities.append({
						"type": entity_type.capitalize(),
						"atom": atom,
						"atom_id": atom_id,
						"x": x,
						"y": y
					})
			else:
				for lexical_grounding in lexical_groundings:
					if lexical_grounding["atom_id"] == atom_id and entity_type.lower() not in entity_types_to_exclude and entity_type.lower() not in rooms:
						important_entities.append({
							"type": entity_type.capitalize(),
							"atom": lexical_grounding["atom"],
							"atom_id": lexical_grounding["atom_id"],
							"token_id": lexical_grounding["token_id"],
							"x": x,
							"y": y
						})
						break

		distance_limit = max(2.0, math.ceil(float(math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2))/10))

		for entity in important_entities:
			min_dist = 1000
			min_idx = -1
			x_1 = entity["x"]
			y_1 = entity["y"]
			for idx, huric_room in enumerate(huric_rooms):
				x_2 = huric_room["x"]
				y_2 = huric_room["y"]

				distance = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
				# print(f"Distance between entity {entity['type']} and room {huric_room['room']}: {distance}")
				if distance <= distance_limit and distance < min_dist:
					min_dist = distance
					min_idx = idx
			if min_idx != -1:
				spatial_relations.append(f"{entity['type']} is in the {huric_rooms[min_idx]['room']}")
		

		# Generate entity subsets
		for i in range(0, len(important_entities) + 1):
			for subset in itertools.combinations(important_entities, i):
				to_include = list(subset)
				to_exclude = [entity for entity in important_entities if entity not in to_include]
				set_to_append = {
					"to_include": to_include,
					"to_exclude": to_exclude,
				}
				entity_subsets.append(set_to_append)

		# Extract spatial relations
		for entity_1 in important_entities:
			x_1 = entity_1["x"]
			y_1 = entity_1["y"]

			if entity_1['atom_id'] not in far_relation_dict:
				far_relation_dict[entity_1['atom_id']] = []
			
			if entity_1['atom'] not in close_relation_dict:
				close_relation_dict[entity_1['atom']] = []

			for entity_2 in important_entities:
				if entity_1['atom'] == entity_2['atom']:
					continue
				x_2 = entity_2["x"]
				y_2 = entity_2["y"]

				distance = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
				# print(f"Distance between entity {entity_1['type']} and entity {entity_2['type']}: {distance}")
				if distance <= distance_limit:
					close_relation_dict[entity_1['atom']].append(entity_2)
					spatial_relations.append(f"{entity_1['type']} is close to {entity_2['type']}")
				else:
					far_relation_dict[entity_1['atom_id']].append(entity_2['atom_id'])
					spatial_relations.append(f"{entity_1['type']} is far from {entity_2['type']}")
		
		# Extract frame semantics
		for item in root.find("commands").find("command").find("semantics").find("frames").findall("frame"):
			frame_name = item.attrib['name']
			lex_unit = item.find("lexicalUnit").find("token").attrib['id']
			lemma_frame_name = tokens[int(lex_unit) - 1]
			frame_elements = item.find("frameElements").findall("frameElement")
			frame_phrase = f"Frame: \"{frame_name}\" (evoked in the command by \"{lemma_frame_name}\")\n\t\t•\t"
			frame_element_phrases = []
			for frame_element in frame_elements:
				frame_element_type = frame_element.attrib['type']
				if 'semanticHead' not in frame_element.attrib:
					frame_element_semantic_head = frame_element.find("token").attrib['id']
				else:
					frame_element_semantic_head = frame_element.attrib['semanticHead']
				frame_element_semantic_head_word = None
				for important_entity in important_entities:
					if frame_element_semantic_head == important_entity["token_id"]:
						frame_element_semantic_head_word = important_entity["type"]
						break
				if frame_element_semantic_head_word is None:
					if int(frame_element_semantic_head) > len(tokens):
						# print("Warning: Semantic head", frame_element_semantic_head, "is out of bounds for tokens list in huric file", huric_file_path, "so the id is ignored and will be taken the last token id declared inside the Frame Element.")
						frame_element_semantic_head = frame_element.findall("token")[-1].attrib['id']
					frame_element_semantic_head_word = tokens[int(frame_element_semantic_head)-1].capitalize()
				frame_element_lemmas = []
				for t in frame_element.findall("token"):
					if int(t.attrib['id']) <= len(tokens):
						frame_element_lemmas.append(tokens[int(t.attrib['id'])-1])
					else:
						print("Warning: Token not found in tokens list:", t.attrib['id'], "for huric file", huric_file_path, "so the id is ignored.")
				frame_element_phrases.append(f"Frame Element: \"{frame_element_type}\" → \"{frame_element_semantic_head_word}\" (represented in the command by \"{' '.join(frame_element_lemmas)}\").")
				
			frame_phrase = frame_phrase + "\n\t\t•\t".join(frame_element_phrases)
			frame_semantics.append(frame_phrase)
		
		
		# print("Important entities:", [(entity["type"], entity["x"], entity["y"]) for entity in important_entities])
		# print("Optional entities:", optional_entities)
		# print("Spatial relations:", spatial_relations)
		# print("Command:", command)
		# print("Entity subsets:", entity_subsets)
		# print("Far relations:", far_relation_dict)

		# Remove entity subsets that should not exist because they contain entities that are far from each other
		for subset in entity_subsets:
			entities_to_include = subset['to_include']
			if len(subset) > 1:
				for entity_1 in entities_to_include:
					for entity_2 in entities_to_include:
						if entity_1['atom_id'] in far_relation_dict and entity_2['atom_id'] in far_relation_dict[entity_1['atom_id']] and subset in entity_subsets:
							entity_subsets.remove(subset)
							break
		
		# Flag entity subsets that have all entities that are close to each other
		for subset in entity_subsets:
			entities_to_include = subset['to_include']
			if len(entities_to_include) == 0 and len(important_entities) > 0:
				subset['negative_example'] = True
				subset['negative_example_entities'] = important_entities
				subset['negative_example_entities_count'] = len(important_entities)
			else:
				subset['negative_example'] = False
				subset['negative_example_entities'] = []
				subset['negative_example_entities_count'] = 0
			for entity_1 in entities_to_include:
				close_entities = close_relation_dict[entity_1['atom']]
				for entity_2 in entities_to_include:
					if entity_2['atom'] == entity_1['atom']:
						continue
					for close_entity in close_entities:
						if entity_2['atom'] == close_entity['atom']:
							close_entities.remove(close_entity)
				subset['negative_example'] |= len(close_entities) > 0
				subset['negative_example_entities'].extend(close_entities)
				subset['negative_example_entities_count'] += len(close_entities)



		
		# print("Entity subsets:", entity_subsets)
		# print("Far relations:", far_relation_dict)
		# print("Close relations:", close_relation_dict)
		for conf_type in configuration_type:
			write_prompt_files_and_configs_by_type(id_huric, important_entities, optional_entities, command, frame_semantics, spatial_relations, entity_subsets, output_folder, prompt_templates, conf_type)
  
  
def process_huric_file(huric_file_path, output_folder, prompt_template_folder, rooms, entity_types_to_exclude):
	important_entities = []
	optional_entities = []
	huric_rooms = []
	spatial_relations = []
	frame_semantics = []
	
	lexical_groundings = []

	tokens = []

	entity_subsets = []
	far_relation_dict = {}
	close_relation_dict = {}

	prompt_templates = get_prompt_templates(prompt_template_folder)
	os.makedirs(output_folder, exist_ok=True)


	print(f"Analyzing Huric file {huric_file_path}")
	with open(huric_file_path,"r") as f:
		tree = ET.parse(f)
		root = tree.getroot()

		id_huric = root.attrib['id']

		# Extract important entities
		for item in root.find('lexicalGroundings').findall('lexicalGrounding'):
			atom = item.attrib['atom']
			token_id = item.attrib['tokenId']
			lexical_groundings.append({
				"atom": atom,
				"atom_id": atom.split("_")[1],
				"token_id": token_id
			})
		
		# Extract commands
		command_element = root.find('commands').find('command')
		command = command_element.find('sentence').text
		for token_element in command_element.find('tokens').findall('token'):
			tokens.append(token_element.attrib['lemma'])


		# Min and max dimensions to determine the ambient max dimensions
		min_x = 1000
		max_x = -1000
		min_y = 1000
		max_y = -1000
		# Extract optional entities
		for item in root.find("semanticMap").find("entities").findall("entity"):
			atom = item.attrib['atom']
			entity_type = item.attrib['type']
			atom_id = atom.split("_")[1]
			coord = item.find('coordinate')
			x = float(coord.attrib['x'])
			y = float(coord.attrib['y'])

			# Extract ambient max dimensions
			if x < min_x:
				min_x = x
			if x > max_x:
				max_x = x
			if y < min_y:
				min_y = y
			if y > max_y:
				max_y = y

			if entity_type.lower() in rooms:
				print(f"Room found: {entity_type}")
				huric_room = {
					"room": entity_type.capitalize(),
					"x": x,
					"y": y
				}
				huric_rooms.append(huric_room)
			elif atom_id not in [lexical_grounding["atom_id"] for lexical_grounding in lexical_groundings]:
				if entity_type.lower() not in entity_types_to_exclude and entity_type.lower() not in rooms:
					optional_entities.append({
						"type": entity_type.capitalize(),
						"atom": atom,
						"atom_id": atom_id,
						"x": x,
						"y": y
					})
			else:
				for lexical_grounding in lexical_groundings:
					if lexical_grounding["atom_id"] == atom_id and entity_type.lower() not in entity_types_to_exclude and entity_type.lower() not in rooms:
						important_entities.append({
							"type": entity_type.capitalize(),
							"atom": lexical_grounding["atom"],
							"atom_id": lexical_grounding["atom_id"],
							"token_id": lexical_grounding["token_id"],
							"x": x,
							"y": y
						})
						break

		distance_limit = max(2.0, math.ceil(float(math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2))/10))

		for entity in important_entities:
			min_dist = 1000
			min_idx = -1
			x_1 = entity["x"]
			y_1 = entity["y"]
			for idx, huric_room in enumerate(huric_rooms):
				x_2 = huric_room["x"]
				y_2 = huric_room["y"]

				distance = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
				print(f"Distance between entity {entity['type']} and room {huric_room['room']}: {distance}")
				if distance <= distance_limit and distance < min_dist:
					min_dist = distance
					min_idx = idx
			if min_idx != -1:
				spatial_relations.append(f"{entity['type']} is in the {huric_rooms[min_idx]['room']}")
		

		# Generate entity subsets
		for i in range(0, len(important_entities) + 1):
			for subset in itertools.combinations(important_entities, i):
				to_include = list(subset)
				to_exclude = [entity for entity in important_entities if entity not in to_include]
				set_to_append = {
					"to_include": to_include,
					"to_exclude": to_exclude,
				}
				entity_subsets.append(set_to_append)

		# Extract spatial relations
		for entity_1 in important_entities:
			x_1 = entity_1["x"]
			y_1 = entity_1["y"]

			if entity_1['atom_id'] not in far_relation_dict:
				far_relation_dict[entity_1['atom_id']] = []
			
			if entity_1['atom'] not in close_relation_dict:
				close_relation_dict[entity_1['atom']] = []

			for entity_2 in important_entities:
				if entity_1['atom'] == entity_2['atom']:
					continue
				x_2 = entity_2["x"]
				y_2 = entity_2["y"]

				distance = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
				print(f"Distance between entity {entity_1['type']} and entity {entity_2['type']}: {distance}")
				if distance <= distance_limit:
					close_relation_dict[entity_1['atom']].append(entity_2)
					spatial_relations.append(f"{entity_1['type']} is close to {entity_2['type']}")
				else:
					far_relation_dict[entity_1['atom_id']].append(entity_2['atom_id'])
					spatial_relations.append(f"{entity_1['type']} is far from {entity_2['type']}")
		
		# Extract frame semantics
		for item in root.find("commands").find("command").find("semantics").find("frames").findall("frame"):
			frame_name = item.attrib['name']
			lex_unit = item.find("lexicalUnit").find("token").attrib['id']
			lemma_frame_name = tokens[int(lex_unit) - 1]
			frame_elements = item.find("frameElements").findall("frameElement")
			frame_phrase = f"Frame: \"{frame_name}\" (evoked in the command by \"{lemma_frame_name}\")\n\t\t•\t"
			frame_element_phrases = []
			for frame_element in frame_elements:
				frame_element_type = frame_element.attrib['type']
				if 'semanticHead' not in frame_element.attrib:
					frame_element_semantic_head = frame_element.find("token").attrib['id']
				else:
					frame_element_semantic_head = frame_element.attrib['semanticHead']
				frame_element_semantic_head_word = None
				for important_entity in important_entities:
					if frame_element_semantic_head == important_entity["token_id"]:
						frame_element_semantic_head_word = important_entity["type"]
						break
				if frame_element_semantic_head_word is None:
					if int(frame_element_semantic_head) > len(tokens):
						print("Warning: Semantic head", frame_element_semantic_head, "is out of bounds for tokens list in huric file", huric_file_path, "so the id is ignored and will be taken the last token id declared inside the Frame Element.")
						frame_element_semantic_head = frame_element.findall("token")[-1].attrib['id']
					frame_element_semantic_head_word = tokens[int(frame_element_semantic_head)-1].capitalize()
				frame_element_lemmas = []
				for t in frame_element.findall("token"):
					if int(t.attrib['id']) <= len(tokens):
						frame_element_lemmas.append(tokens[int(t.attrib['id'])-1])
					else:
						print("Warning: Token not found in tokens list:", t.attrib['id'], "for huric file", huric_file_path, "so the id is ignored.")
				frame_element_phrases.append(f"Frame Element: \"{frame_element_type}\" → \"{frame_element_semantic_head_word}\" (represented in the command by \"{' '.join(frame_element_lemmas)}\").")
				
			frame_phrase = frame_phrase + "\n\t\t•\t".join(frame_element_phrases)
			frame_semantics.append(frame_phrase)
		
		
		print("Important entities:", [(entity["type"], entity["x"], entity["y"]) for entity in important_entities])
		print("Optional entities:", optional_entities)
		print("Spatial relations:", spatial_relations)
		print("Command:", command)
		print("Entity subsets:", entity_subsets)
		print("Far relations:", far_relation_dict)

		# Remove entity subsets that should not exist because they contain entities that are far from each other
		for subset in entity_subsets:
			entities_to_include = subset['to_include']
			if len(subset) > 1:
				for entity_1 in entities_to_include:
					for entity_2 in entities_to_include:
						if entity_1['atom_id'] in far_relation_dict and entity_2['atom_id'] in far_relation_dict[entity_1['atom_id']] and subset in entity_subsets:
							entity_subsets.remove(subset)
							break
		
		# Flag entity subsets that have all entities that are close to each other
		for subset in entity_subsets:
			entities_to_include = subset['to_include']
			if len(entities_to_include) == 0 and len(important_entities) > 0:
				subset['negative_example'] = True
				subset['negative_example_entities'] = important_entities
				subset['negative_example_entities_count'] = len(important_entities)
			else:
				subset['negative_example'] = False
				subset['negative_example_entities'] = []
				subset['negative_example_entities_count'] = 0
			for entity_1 in entities_to_include:
				close_entities = close_relation_dict[entity_1['atom']]
				for entity_2 in entities_to_include:
					if entity_2['atom'] == entity_1['atom']:
						continue
					for close_entity in close_entities:
						if entity_2['atom'] == close_entity['atom']:
							close_entities.remove(close_entity)
				subset['negative_example'] |= len(close_entities) > 0
				subset['negative_example_entities'].extend(close_entities)
				subset['negative_example_entities_count'] += len(close_entities)



		
		print("Entity subsets:", entity_subsets)
		print("Far relations:", far_relation_dict)
		print("Close relations:", close_relation_dict)
		write_prompt_files_and_configs(id_huric, important_entities, optional_entities, command, frame_semantics, spatial_relations, entity_subsets, output_folder, prompt_templates)


def get_prompt_templates(prompt_template_folder):
	with open(f"{prompt_template_folder}template_full.txt","r", encoding="utf8") as f:
		template_full = f.read()
	with open(f"{prompt_template_folder}template_partial.txt","r", encoding="utf8") as f:
		template_partial = f.read()
	with open(f"{prompt_template_folder}template_empty.txt","r", encoding="utf8") as f:
		template_empty = f.read()
	return {"full": template_full, "partial": template_partial, "empty": template_empty}


def write_prompt_file(idx, prompt_type, id_huric, important_entities_to_include, important_entities_to_exclude, optional_entities, command, frame_semantics, spatial_relations, entity_subset, output_folder, prompt_templates):
	prompt_copy = prompt_templates[prompt_type]
	optional_entities_names = [entity["type"] for entity in optional_entities]

	if len(important_entities_to_include) > 0:
		prompt_copy = prompt_copy.replace("<IMPORTANT_ENTITIES_TO_INCLUDE>", ", ".join(important_entities_to_include))
	else:
		prompt_copy = prompt_copy.replace("<IMPORTANT_ENTITIES_TO_INCLUDE>", "None")
	
	if len(important_entities_to_exclude) > 0:
		prompt_copy = prompt_copy.replace("<IMPORTANT_ENTITIES_TO_EXCLUDE>", "Person, Robot, " + ", ".join(important_entities_to_exclude))
	else:
		prompt_copy = prompt_copy.replace("<IMPORTANT_ENTITIES_TO_EXCLUDE>", "Person, Robot")
	
	if len(optional_entities) > 0:
		prompt_copy = prompt_copy.replace("<OPTIONAL_ENTITIES>", ", ".join(optional_entities_names))
	else:
		prompt_copy = prompt_copy.replace("<OPTIONAL_ENTITIES>", "None")

	if len(command) > 0:
		prompt_copy = prompt_copy.replace("<COMMAND>", f"\"{command}\"")
	else:
		prompt_copy = prompt_copy.replace("<COMMAND>", "None")

	if len(frame_semantics) > 0:
		prompt_copy = prompt_copy.replace("<FRAME_SEMANTICS>", "•\t" + "\n\t•	".join(frame_semantics))
	else:
		prompt_copy = prompt_copy.replace("<FRAME_SEMANTICS>", "•\tNone")

	if len(spatial_relations) > 0:
		prompt_copy = prompt_copy.replace("<SPATIAL_RELATIONS>","\n\t•\t".join(spatial_relations))
	else:
		prompt_copy = prompt_copy.replace("<SPATIAL_RELATIONS>", "None")

	with open(f"{output_folder}{id_huric}-{prompt_type}-{idx}.txt", "w", encoding="utf8") as f_out:
		f_out.write(prompt_copy)
	
	write_config(idx, prompt_type, id_huric, important_entities_to_include, important_entities_to_exclude , optional_entities, command, frame_semantics, spatial_relations, entity_subset, output_folder, prompt_templates)


def write_config(idx, prompt_type, id_huric, important_entities_to_include, important_entities_to_exclude, optional_entities, command, frame_semantics, spatial_relations, entity_subset, output_folder, prompt_templates):
	with open(f"{output_folder}{id_huric}-{prompt_type}-{idx}_config.json", "w", encoding="utf8") as f_out:
		json.dump({
			"prompt_type": prompt_type,
			"optional_entities": optional_entities,
			"command": command,
			"frame_semantics": frame_semantics,
			"spatial_relations": spatial_relations,
			"important_entities_to_include": entity_subset["to_include"],
			"important_entities_to_exclude": entity_subset["to_exclude"],
			"negative_example": entity_subset["negative_example"],
			"negative_example_entities": entity_subset["negative_example_entities"],
			"negative_example_entities_count": entity_subset["negative_example_entities_count"],
		}, f_out)


def write_prompt_files_and_configs(id_huric, important_entities, optional_entities, command, frame_semantics, spatial_relations, entity_subsets, output_folder, templates):
	for idx, entity_subset in enumerate(entity_subsets):
		important_entities_to_include = [entity["type"] for entity in entity_subset["to_include"]]
		important_entities_to_exclude = [entity["type"] for entity in entity_subset["to_exclude"]]

		if len(important_entities_to_include) == 0:
			write_prompt_file(idx, "empty", id_huric, important_entities_to_include, important_entities_to_exclude, optional_entities, command, frame_semantics, spatial_relations, entity_subset, output_folder, templates)
		elif len(important_entities_to_include) == len(important_entities):
			write_prompt_file(idx, "full", id_huric, important_entities_to_include, important_entities_to_exclude, optional_entities, command, frame_semantics, spatial_relations, entity_subset, output_folder, templates)
		else:
			write_prompt_file(idx, "partial", id_huric, important_entities_to_include, important_entities_to_exclude, optional_entities, command, frame_semantics, spatial_relations, entity_subset, output_folder, templates)


def write_prompt_files_and_configs_by_type(id_huric, important_entities, optional_entities, command, frame_semantics, spatial_relations, entity_subsets, output_folder, templates, configuration_type):
	for idx, entity_subset in enumerate(entity_subsets):
		important_entities_to_include = [entity["type"] for entity in entity_subset["to_include"]]
		important_entities_to_exclude = [entity["type"] for entity in entity_subset["to_exclude"]]

		write_prompt_file(idx, configuration_type, id_huric, important_entities_to_include, important_entities_to_exclude, optional_entities, command, frame_semantics, spatial_relations, entity_subset, output_folder, templates)

def main():
	parser = argparse.ArgumentParser("main.py")
	parser.add_argument("--config", help="The config file for the execution.", type=str, default='config.ini')
	args = parser.parse_args()
	config_path = args.config

	config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')] if len(x) > 0 else []})
	config.read(config_path)

	huric_ids = config['Execution'].getlist('huric_ids')

	if len(huric_ids) == 0:
		raise Exception("No huric ids provided in config file. Provide specific ids separated by a comma to process only specific huric files or 'all' to process all huric files.")
	print(huric_ids)

	process_all = False

	if 'all' in huric_ids:
		process_all = True
	
	rooms = config['Entities'].getlist('rooms')
	entity_types_to_exclude = config['Entities'].getlist('entity_types_to_exclude')
	output_folder = config['Execution'].get('output_folder')
	prompt_template_folder = config['Execution'].get('prompt_template_folder')
	huric_folder = config['Execution'].get('huric_folder')
	huric_files = glob(f"{huric_folder}**/**.hrc")
	huric_files_to_process = []

	for huric_id in huric_ids:
		for f in huric_files:
			if process_all or huric_id in f:
				huric_files_to_process.append(f.replace("\\", "/"))
	print(huric_files_to_process)

	for huric_file in huric_files_to_process:
		process_huric_file(huric_file, output_folder, prompt_template_folder, rooms, entity_types_to_exclude)

	return


if __name__ == "__main__":
	main()