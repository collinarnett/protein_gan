from pathlib import Path
from urllib import request
from tqdm import tqdm

test_folder = Path("test")
train_folder = Path("train")
test_list = Path('test_ids.txt').read_text().splitlines()
train_list = Path('train_ids.txt').read_text().splitlines()

test_folder.mkdir(exist_ok=True)
train_folder.mkdir(exist_ok=True)

for ids in tqdm(test_list):
    p = test_folder / f'{ids}.pdb.gz'
    request.urlretrieve(f"http://files.rcsb.org/download/{ids}.pdb.gz", p)
for ids in tqdm(train_list):
    p = train_folder / f'{ids}.pdb.gz'
    request.urlretrieve(f"http://files.rcsb.org/download/{ids}.pdb.gz", p)
