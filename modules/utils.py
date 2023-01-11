import os
from typing import Optional


def search_ext_file(path: str, ext: str) -> Optional[str]:
    files = os.listdir(path)
    for f in files:
        if f.endswith(ext):
            return os.path.join(path, f)
    return None
