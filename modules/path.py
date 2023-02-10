import os
import sys

webui_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, webui_path)

paths = [
    {
        "t": 0,
        "p": os.path.join(webui_path, "repositories/sovits")
    }
]


def insert_repositories_path():
    for p in paths:
        if p["t"] == 0:
            sys.path.insert(0, p["p"])
        else:
            sys.path.append(p["p"])


insert_repositories_path()
