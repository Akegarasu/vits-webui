import io
import re
import time
from typing import Tuple, List

import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile
from torch import no_grad, LongTensor

import modules.sovits_model as sovits_model
import modules.vits_model as vits_model
from modules.devices import device, torch_gc
from modules.sovits_model import Svc as SovitsSvc
from modules.utils import windows_filename
from modules.vits_model import VITSModel
from vits import commons
from vits.text import text_to_sequence


class Text2SpeechTask:
    origin: str
    speaker: str
    method: str
    pre_processed: List[Tuple[int, str]]

    def __init__(self, origin: str, speaker: str, method: str):
        self.origin = origin
        self.speaker = speaker
        self.method = method
        self.pre_processed = []

    def preprocess(self):
        model = vits_model.curr_vits_model
        if self.method == "Simple":
            speaker_id = model.speakers.index(self.speaker)
            self.pre_processed.append((speaker_id, self.origin))
        elif self.method == "Multi Speakers":
            match = re.findall(r"\[(.*)] (.*)", self.origin)
            for m in match:
                if m[0] not in model.speakers:
                    err = f"Error: Unknown speaker {m[0]}, check your input."
                    print(err)
                    return err
                speaker_id = model.speakers.index(m[0])
                self.pre_processed.append((speaker_id, m[1]))
        elif self.method == "Batch Process":
            spl = self.origin.split("\n")
            speaker_id = model.speakers.index(self.speaker)
            for line in spl:
                self.pre_processed.append((speaker_id, line))


class SovitsTask:
    # 暂时貌似没有需要 preprocess 的，就先放这里了
    pass


def text2speech(text: str, speaker: str, speed, method="Simple"):
    task = Text2SpeechTask(origin=text, speaker=speaker, method=method)
    err = task.preprocess()
    if err:
        return err, None
    ti = int(time.time())
    save_path = ""
    output_info = "Success saved to "
    outputs = []
    for t in task.pre_processed:
        sample_rate, data = process_vits(model=vits_model.curr_vits_model,
                                         text=t[1], speaker_id=t[0], speed=speed)
        outputs.append(data)
        save_path = f"outputs/vits/{str(ti)}-{windows_filename(t[1])}.wav"
        wavfile.write(save_path, sample_rate, data)
        output_info += f"\n{save_path}"
        ti += 1

    torch_gc()

    if len(outputs) > 1:
        batch_file_path = f"outputs/vits-batch/{str(int(time.time()))}.wav"
        wavfile.write(batch_file_path, vits_model.curr_vits_model.hps.data.sampling_rate, np.concatenate(outputs))
        return f"{output_info}\n{batch_file_path}", batch_file_path
    return output_info, save_path


def sovits_process(audio, speaker: str, vc_transform: int):
    return process_so_vits(svc_model=sovits_model.curr_sovits_model,
                           sid=speaker,
                           input_audio=audio,
                           vc_transform=vc_transform)


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


def process_vits(model: VITSModel, text: str,
                 speaker_id, speed,
                 noise_scale=0.667,
                 noise_scale_w=0.8) -> Tuple[int, np.array]:
    stn_tst = text_processing(text, model)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.model.infer(x_tst, x_tst_lengths, sid=sid,
                                  noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                  length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    return model.hps.data.sampling_rate, audio


def process_so_vits(svc_model: SovitsSvc, sid, input_audio, vc_transform):
    if input_audio is None:
        return "You need to input an audio", None
    sampling_rate, audio = input_audio
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    print(audio.shape)
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, audio, 16000, format="wav")
    out_wav_path.seek(0)

    out_audio, out_sr = svc_model.infer(sid, vc_transform, out_wav_path)
    _audio = out_audio.cpu().numpy()
    return "Success", (32000, _audio)
