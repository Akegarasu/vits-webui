import hashlib
import os
import time
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import parselmouth
import soundfile
import torch
import torchaudio

import modules.utils as utils
from modules.devices import device
from modules.model import ModelInfo
from modules.model import refresh_model_list
from modules.utils import model_hash
from repositories.sovits.hubert import hubert_model
from repositories.sovits.models import SynthesizerTrn

MODEL_PATH = os.path.join(os.path.join(os.getcwd(), "models"), "sovits")


class Svc(object):
    def __init__(self, net_g_path, config_path, hubert_path="models/hubert/hubert-soft-0d54a1f4.pt"):
        self.net_g_path = net_g_path
        self.hubert_path = hubert_path
        self.dev = device
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.speakers = {}
        for spk, sid in self.hps_ms.spk.items():
            self.speakers[sid] = spk
        self.spk2id = self.hps_ms.spk

    def load_model(self):
        self.hubert_soft = hubert_model.hubert_soft(self.hubert_path)
        if self.dev != torch.device("cpu"):
            self.hubert_soft = self.hubert_soft.cuda()
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model)
        _ = load_checkpoint(self.net_g_path, self.net_g_ms, None)
        if "half" in self.net_g_path and self.dev != torch.device("cpu"):
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)

    def get_units(self, source, sr):
        source = source.unsqueeze(0).to(self.dev)
        with torch.inference_mode():
            start = time.time()
            units = self.hubert_soft.units(source)
            use_time = time.time() - start
            print("hubert use time:{}".format(use_time))
            return units

    def get_unit_pitch(self, in_path, tran):
        source, sr = torchaudio.load(in_path)
        source = torchaudio.functional.resample(source, sr, 16000)
        if len(source.shape) == 2 and source.shape[1] >= 2:
            source = torch.mean(source, dim=0).unsqueeze(0)
        soft = self.get_units(source, sr).squeeze(0).cpu().numpy()
        f0_coarse, f0 = get_f0(source.cpu().numpy()[0], soft.shape[0] * 2, tran)
        return soft, f0

    def infer(self, speaker_id, tran, raw_path):
        if type(speaker_id) == str:
            speaker_id = self.spk2id[speaker_id]
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)
        soft, pitch = self.get_unit_pitch(raw_path, tran)
        f0 = torch.FloatTensor(clean_pitch(pitch)).unsqueeze(0).to(self.dev)
        if "half" in self.net_g_path and torch.cuda.is_available():
            stn_tst = torch.HalfTensor(soft)
        else:
            stn_tst = torch.FloatTensor(soft)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.dev)
            start = time.time()
            x_tst = torch.repeat_interleave(x_tst, repeats=2, dim=1).transpose(1, 2)
            audio = self.net_g_ms.infer(x_tst, f0=f0, g=sid)[0, 0].data.float()
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1]


class SovitsModel(Svc):
    def __init__(self, info: ModelInfo):
        super(SovitsModel, self).__init__(info.checkpoint_path, info.config_path)
        self.model_name = info.model_name


sovits_model_list: Dict[str, ModelInfo] = {}
curr_sovits_model: SovitsModel = None


def get_model() -> Svc:
    return curr_sovits_model


def get_model_name():
    return curr_sovits_model.model_name if curr_sovits_model is not None else None


def get_model_list():
    return [k for k, _ in sovits_model_list.items()]


def get_speakers():
    if curr_sovits_model is None:
        return ["None"]
    return [spk for sid, spk in curr_sovits_model.speakers.items() if spk != "None"]


def refresh_list():
    sovits_model_list.clear()
    model_list = refresh_model_list(model_path=MODEL_PATH)
    for m in model_list:
        d = m["dir"]
        p = os.path.join(MODEL_PATH, m["dir"])
        pth_path = m["pth"]
        config_path = m["config"]
        sovits_model_list[d] = ModelInfo(
            model_name=d,
            model_folder=p,
            model_hash=model_hash(pth_path),
            checkpoint_path=pth_path,
            config_path=config_path
        )
    if len(sovits_model_list.items()) == 0:
        print("No so-vits model found. Please put a model in models/sovits")


# def init_model():
#     global curr_sovits_model
#     info = next(iter(sovits_model_list.values()))
#     # load_model(info.model_name)
#     curr_sovits_model = SovitsModel(info)


def load_model(model_name: str):
    global curr_sovits_model, sovits_model_list
    if curr_sovits_model and curr_sovits_model.model_name == model_name:
        return
    info = sovits_model_list[model_name]
    print(f"Loading sovits weights [{info.model_hash}] from {info.checkpoint_path}...")
    m = SovitsModel(info)
    m.load_model()
    curr_sovits_model = m
    print("Sovits model loaded.")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if iteration is None:
        iteration = 1
    if learning_rate is None:
        learning_rate = 0.0002
    if optimizer is not None and checkpoint_dict['optimizer'] is not None:
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
            # logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


def get_f0(x, p_len, f0_up_key=0):
    time_step = 160 / 16000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    if len(f0) > p_len:
        f0 = f0[:p_len]
    pad_size = (p_len - len(f0) + 1) // 2
    if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0


def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def plt_pitch(input_pitch):
    input_pitch = input_pitch.astype(float)
    input_pitch[input_pitch == 1] = np.nan
    return input_pitch


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])
