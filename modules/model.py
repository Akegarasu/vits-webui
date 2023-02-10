import os
from modules.utils import search_ext_file


class ModelInfo:
    model_name: str
    model_folder: str
    model_hash: str
    checkpoint_path: str
    config_path: str

    def __init__(self, model_name, model_folder, model_hash, checkpoint_path, config_path):
        self.model_name = model_name
        self.model_folder = model_folder
        self.model_hash = model_hash
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.custom_symbols = None


def refresh_model_list(model_path):
    dirs = os.listdir(model_path)
    models = []
    for d in dirs:
        p = os.path.join(model_path, d)
        if not os.path.isdir(p):
            continue
        pth_path = search_ext_file(p, ".pth")
        if not pth_path:
            print(f"Path {p} does not have a pth file, pass")
            continue
        config_path = search_ext_file(p, ".json")
        if not config_path:
            print(f"Path {p} does not have a config file, pass")
            continue
        models.append({
            "dir": d,
            "pth": pth_path,
            "config": config_path
        })
    return models
