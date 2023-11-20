import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer


def prepare(
    destination_path: Path = Path("data/text"),
    checkpoint_dir: Path = Path("checkpoints/EleutherAI/pythia-70M"),
    seed: int = 42,
    test_size: Union[float, int, None] = 0.2,  # Adjust the test size as needed
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)
    text_data = "data/text/privacydata.txt"
    tokenizer = Tokenizer(checkpoint_dir)

    # number of workers in .map() call
    # a good number to use is ~order number of CPU cores // 2
    num_proc = os.cpu_count() // 2

    # Load the text file
    with open(text_data, "r", encoding="utf-8") as f:
        text_data = f.read()

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(text_data, test_size=test_size, random_state=seed)
    train_data = str(train_data)
    print("A")
    train_tokenized = tokenizer.encode(train_data)
    print("b")
    test_tokenized = tokenizer.encode(test_data)
    print("A")
    save_to_binary(train_tokenized, destination_path / "train_data.bin")
    save_to_binary(test_tokenized, destination_path / "test_data.bin")

def save_to_binary(data, filename):
    with open(filename, "wb") as f:
        f.write(data)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
