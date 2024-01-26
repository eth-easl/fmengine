import os
import json

def write_loader_status(ckpt_path: str, skip: int):
    with open(os.path.join(ckpt_path, "dataloader_status.json"), "w") as f:
        json.dump({"skip": skip}, f)