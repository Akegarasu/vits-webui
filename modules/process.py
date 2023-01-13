import time
from typing import Tuple
import scipy.io.wavfile as wavfile
import numpy as np
from torch import no_grad, LongTensor

import modules.vits_model as vits_model
from modules.devices import device, torch_gc
from modules.vits_model import VITSModel
from vits import commons
from vits.text import text_to_sequence


def text2speech(text: str, speaker, speed):
    speaker_id = vits_model.curr_vits_model.speakers.index(speaker)
    sample_rate, data = process_vits(model=vits_model.curr_vits_model,
                                     text=text, speaker_id=speaker_id, speed=speed)
    torch_gc()
    save_path = f"outputs/vits/{str(int(time.time()))}.wav"
    wavfile.write(save_path, sample_rate, data)
    return "Success", save_path


def text_processing(text, model: VITSModel):
    hps = model.hps
    _use_symbols = model.symbols
    # 留了点屎山 以后再处理吧
    if hasattr(hps, "symbols_zh"):
        _use_symbols = hps.symbols_zh
    text_norm = text_to_sequence(text, _use_symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def process_vits(model: VITSModel, text: str, speaker_id, speed) -> Tuple[int, np.array]:
    stn_tst = text_processing(text, model)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                  length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    return model.hps.data.sampling_rate, audio
