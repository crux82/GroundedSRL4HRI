import json

# Utility per caricare gli ID
def load_ids(filename):
    with open(filename, 'r') as f:
        return set(line.strip() for line in f)

train_ids = load_ids("id_train.txt")
dev_ids = load_ids("id_dev.txt")
test_ids = load_ids("id_test.txt")

filename="json_top_1_images"
directory="./dataset_silver/"
directory_target="./dataset_silver_split/"

with open(directory+filename+".json", "r") as f:
    examples = json.load(f)

train_data, dev_data, test_data = [], [], []
missing=0
for ex in examples:
    id_str = ex["id"].split("_")[0]  
    if id_str in train_ids:
        train_data.append(ex)
    elif id_str in dev_ids:
        dev_data.append(ex)
    elif id_str in test_ids:
        test_data.append(ex)
    else:
        missing += 1
        print(f"⚠️ ID: {id_str}")

with open(directory_target+filename+"_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open(directory_target+filename+"_dev.json", "w") as f:
    json.dump(dev_data, f, indent=2)

with open(directory_target+filename+"_test.json", "w") as f:
    json.dump(test_data, f, indent=2)
print(filename)
print(f"  - train.json: {len(train_data)} examples")
print(f"  - dev.json:   {len(dev_data)} examples")
print(f"  - test.json:  {len(test_data)} examples")
if missing > 0:
    print(f"⚠️  {missing} missing")
print("✅ train.json, dev.json, test.json")
