import json
import os

from datasets import Dataset, DatasetDict

# Get the absolute path to data.json in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data.json")

data = json.load(open(data_path))
dataset: DatasetDict = Dataset.from_list(data).train_test_split(test_size=0.2)

dataset.push_to_hub("ljt019/twenty-questions-600")
