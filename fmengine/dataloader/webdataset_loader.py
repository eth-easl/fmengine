import webdataset as wds
from huggingface_hub import HfFolder
from typing import Dict


def get_wds_dataset(dataset_name_or_path, args: Dict = {}):
    hf_token = HfFolder().get_token()
    dataset = wds.WebDataset(
        f"pipe:curl -s -L https://huggingface.co/datasets/username/my_wds_dataset/resolve/main/train-000000.tar -H 'Authorization:Bearer {hf_token}'"
    )
