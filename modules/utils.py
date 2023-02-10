import os
import platform
import re
import subprocess
from typing import Optional

# for export
from vits.utils import get_hparams_from_file, HParams


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


def open_folder(f):
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist.')
        return
    elif not os.path.isdir(f):
        return

    path = os.path.normpath(f)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        subprocess.Popen(["wsl-open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def windows_filename(s: str):
    return re.sub('[<>:"\/\\|?*\n\t\r]+', "", s)
