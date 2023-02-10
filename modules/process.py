import io
import os.path
import re
import shutil
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
from repositories.sovits.inference import slicer
from pathlib import Path


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
        sample_rate, data = process_vits(model=vits_model.get_model(),
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


def sovits_process(audio_path, speaker: str, vc_transform: int, slice_db: int):
    ti = int(time.time())
    data, sampling_rate = process_so_vits(svc_model=sovits_model.get_model(),
                                          sid=speaker,
                                          input_audio=audio_path,
                                          vc_transform=vc_transform,
                                          slice_db=slice_db)
    save_path = f"outputs/sovits/{str(ti)}.wav"
    soundfile.write(save_path, data, sampling_rate, format="wav")
    torch_gc()
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


def process_so_vits(svc_model: SovitsSvc, sid, input_audio, vc_transform, slice_db):
    if input_audio is None:
        return "You need to input an audio", None

    audio_path = input_audio.name
    wav_path = os.path.join("temp", str(int(time.time())) + ".wav")
    if Path(audio_path).suffix != '.wav':
        raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
        soundfile.write(wav_path, raw_audio, raw_sample_rate)
    else:
        shutil.copy(audio_path, wav_path)
    chunks = slicer.cut(wav_path, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

    audio = []
    for (slice_tag, data) in audio_data:
        print(f'segment start, {round(len(data) / audio_sr, 3)}s')
        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        raw_path.seek(0)
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            out_audio, out_sr = svc_model.infer(sid, vc_transform, raw_path)
            _audio = out_audio.cpu().numpy()
        audio.extend(list(_audio))
    os.remove(wav_path)
    return audio, svc_model.target_sample
