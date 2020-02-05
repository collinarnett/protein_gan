from pathlib import Path
from urllib import request

test = 'test_ids.txt'
train = 'train_ids.txt'
test_list = Path(test).read_text().splitlines()
train_list = Path(train).read_text().splitlines()

for ids in test_list:
    request.urlretrieve(f"http://files.rcsb.org/download/{ids}.pdb", f"{ids}.pdb")
