import os
from typing import Optional


def search_ext_file(path: str, ext: str) -> Optional[str]:
    files = os.listdir(path)
    for f in files:
        if f.endswith(ext):
            return os.path.join(path, f)
    return None

def model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'FileNotFound'
