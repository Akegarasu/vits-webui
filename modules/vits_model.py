import importlib.util
import os.path
from typing import List, Dict

import torch

import modules.devices as devices
import vits.utils
from modules.model import ModelInfo, refresh_model_list
from modules.utils import search_ext_file, model_hash
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols as builtin_symbols
from vits.utils import HParams

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

    def __init__(self, info: ModelInfo):
        self.model_name = info.model_name
        self.model_folder = info.model_folder
        self.checkpoint_path = info.checkpoint_path
        self.config_path = info.config_path
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
        model, _, _, _ = load_checkpoint(checkpoint_path=self.checkpoint_path,
                                         model=model, optimizer=None)
        model.eval().to(devices.device)

        self.speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]
        self.model = model
        self.hps = hps
        self.symbols = _symbols

    def load_custom_symbols(self, symbol_path):
        if os.path.exists(symbol_path):
            spec = importlib.util.spec_from_file_location('symbols', symbol_path)
            _sym = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_sym)
            if not hasattr(_sym, "symbols"):
                print(f"Loading symbol file {symbol_path} failed, so such attr")
                return
            self.custom_symbols = _sym


vits_model_list: Dict[str, ModelInfo] = {}
curr_vits_model: VITSModel = None


def get_model() -> VITSModel:
    return curr_vits_model


def get_model_name():
    return curr_vits_model.model_name if curr_vits_model is not None else None


def get_model_list():
    return [k for k, _ in vits_model_list.items()]


def get_speakers():
    return curr_vits_model.speakers if curr_vits_model is not None else ["None"]


def refresh_list():
    vits_model_list.clear()
    model_list = refresh_model_list(model_path=MODEL_PATH)
    for m in model_list:
        d = m["dir"]
        p = os.path.join(MODEL_PATH, m["dir"])
        pth_path = m["pth"]
        config_path = m["config"]
        vits_model_list[d] = ModelInfo(
            model_name=d,
            model_folder=p,
            model_hash=model_hash(pth_path),
            checkpoint_path=pth_path,
            config_path=config_path
        )
    if len(vits_model_list.items()) == 0:
        print("No vits model found. Please put a model in models/vits")


def init_load_model():
    info = next(iter(vits_model_list.values()))
    load_model(info.model_name)


def init_model():
    global curr_vits_model
    info = next(iter(vits_model_list.values()))
    curr_vits_model = VITSModel(info)
    # load_model(info.model_name)


def load_model(model_name: str):
    global curr_vits_model, vits_model_list
    if curr_vits_model and curr_vits_model.model_name == model_name:
        return
    info = vits_model_list[model_name]
    print(f"Loading weights [{info.model_hash}] from {info.checkpoint_path}...")
    m = VITSModel(info)
    m.load_model()
    curr_vits_model = m
    print("Model loaded.")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model, optimizer, learning_rate, iteration
