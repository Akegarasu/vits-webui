import gradio as gr
import os.path
import importlib.util
import vits.utils

from modules.device import device
from vits.models import SynthesizerTrn
from vits.utils import HParams
from vits.text.symbols import symbols as builtin_symbols
from modules.utils import search_ext_file

from typing import List, Dict

# todo: cmdline here
MODEL_PATH = os.path.join(os.path.join(os.getcwd(), "models"), "vits")


class VITSModel:
    model: SynthesizerTrn
    hps: HParams
    symbols: List[str]

    model_name: str
    model_folder: str
    checkpoint_path: str
    config_path: str

    speakers: List[str]

    def __init__(self, model_name: str, model_folder: str, checkpoint_path: str, config_path: str):
        self.model_name = model_name
        self.model_folder = model_folder
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.custom_symbols = None
        # self.state = "" # maybe for multiprocessing

    def load_model(self):
        hps = vits.utils.get_hparams_from_file(self.config_path)
        self.load_custom_symbols(f"{self.model_folder}/symbols.py")
        if self.custom_symbols:
            _symbols = self.custom_symbols.symbols
        elif "symbols" in hps:
            _symbols = hps.symbols
        else:
            _symbols = builtin_symbols

        if hasattr(self.custom_symbols, "symbols_zh"):
            hps["symbols_zh"] = self.custom_symbols.symbols_zh

        model = SynthesizerTrn(
            len(_symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        vits.utils.load_checkpoint(self.checkpoint_path, model, None)
        model.eval().to(device)
        self.speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
        self.speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]

        self.model = model
        self.hps = hps
        self.symbols = _symbols

    # def unload(self):
    #     del self.model

    def load_custom_symbols(self, symbol_path):
        if os.path.exists(symbol_path):
            spec = importlib.util.spec_from_file_location('symbols', symbol_path)
            _sym = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_sym)
            if not hasattr(_sym, "symbols"):
                print(f"Loading symbol file {symbol_path} failed, so such attr")
                return
            self.custom_symbols = _sym
        # todo: custom path


vits_model_list: Dict[str, VITSModel] = {}
curr_vits_model: VITSModel


def get_model() -> VITSModel:
    return curr_vits_model


def get_model_list():
    return [k for k, _ in vits_model_list.items()]


def refresh_list():
    vits_model_list.clear()
    dirs = os.listdir(MODEL_PATH)
    for d in dirs:
        p = os.path.join(MODEL_PATH, d)
        if not os.path.isdir(p):
            continue
        model_path = search_ext_file(p, ".pth")
        if not model_path:
            print(f"Path {p} does not have a pth file, pass")
            continue
        config_path = search_ext_file(p, ".json")
        if not config_path:
            print(f"Path {p} does not have a config file, pass")
            continue
        vits_model_list[d] = VITSModel(
            model_name=d,
            model_folder=p,
            checkpoint_path=model_path,
            config_path=config_path
        )
    if len(vits_model_list.items()) == 0:
        raise "Please put a model in models/vits"


def init_load_model():
    global curr_vits_model, vits_model_list
    m = next(iter(vits_model_list.values()))
    print(f"Loading weights from {m.model_folder}...")
    m.load_model()
    curr_vits_model = m
    print("Model loaded.")


def load_model(model_name: str):
    global curr_vits_model, vits_model_list
    if curr_vits_model.model_name == model_name:
        return
    m = vits_model_list[model_name]
    print(f"Loading weights from {m.model_folder}...")
    m.load_model()
    curr_vits_model = m
    print("Model loaded.")
