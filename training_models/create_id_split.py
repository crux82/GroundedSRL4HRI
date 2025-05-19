import random

random.seed(42)
input_path="../image_generator/data/dataset/text_top_1_images.tsv"

with open(input_path, "r") as f:
    lines = f.readlines()

ids = set()
for line in lines:
    id_part = line.strip().split("_")[0]
    ids.add(id_part)

ids = list(ids)
random.shuffle(ids)

#Split train/dev/test (80/10/10)
n = len(ids)
n_train = int(0.8 * n)
n_dev = int(0.10 * n)
n_test = n - n_train - n_dev

id_train = ids[:n_train]
id_dev = ids[n_train:n_train + n_dev]
id_test = ids[n_train + n_dev:]

with open("id_train.txt", "w") as f:
    f.write("\n".join(id_train))

with open("id_dev.txt", "w") as f:
    f.write("\n".join(id_dev))

with open("id_test.txt", "w") as f:
    f.write("\n".join(id_test))

print("✅ id_train.txt, id_dev.txt, id_test.txt")
