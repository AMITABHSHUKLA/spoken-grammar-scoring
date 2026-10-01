"""Frozen audio models, run once per clip.

    python extraction/audio_features.py --what wavlm   # microsoft/wavlm-large: mean and std of every layer over time
    python extraction/audio_features.py --what ctc     # facebook/wav2vec2-large-960h-lv60-self: share of "blank" frames

WavLM gives the acoustic view (layer 20) and the voice vectors used to group speakers (layers 3 and 6).
The CTC model labels every 20 ms frame with a letter or with "blank"; a clip that is almost all blank
contains no speech. It has no language model, so it cannot invent words the way Whisper does.

Outputs in --out: wavlm.npz (split, filename, mean[N, layers, dim], std[N, layers, dim], float16) and ctc.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR = 16000


def list_clips(data_dir):
    rows = []
    for split in ("train", "test"):
        rows += [(split, f) for f in pd.read_csv(data_dir / f"{split}.csv")["filename"]]
    return rows


def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path}: unexpected sample rate {sr}"
    return audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)


def windows(audio, seconds, min_seconds=1.0):
    step = int(seconds * SR)
    chunks = [audio[i:i + step] for i in range(0, len(audio), step)]
    return [c for c in chunks if len(c) >= min_seconds * SR] or [audio]


class PooledStats:
    """Running mean/std over frames, per layer, across all windows of one clip."""

    def __init__(self):
        self.s = self.ss = None
        self.n = 0

    def add(self, hidden):  # hidden: [layers, frames, dim] float32 on GPU
        h = hidden.double()
        s, ss = h.sum(1), (h * h).sum(1)
        self.s = s if self.s is None else self.s + s
        self.ss = ss if self.ss is None else self.ss + ss
        self.n += hidden.shape[1]

    def result(self):
        mean = self.s / self.n
        std = (self.ss / self.n - mean * mean).clamp_min(0).sqrt()
        return mean.float().cpu().numpy(), std.float().cpu().numpy()


def load_wavlm(model_id, device):
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = WavLMModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    return fe, model


def load_ctc(model_id, device):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    return proc, model


@torch.no_grad()
def wavlm_clip(audio, fe, model, device):
    stats = PooledStats()
    for w in windows(audio, 20):
        x = fe(w, sampling_rate=SR, return_tensors="pt").input_values.to(device, torch.float16)
        out = model(x, output_hidden_states=True)
        stats.add(torch.stack(out.hidden_states)[:, 0].float())
    return stats.result()


@torch.no_grad()
def ctc_clip(audio, proc, model, device):
    texts, maxp, blank_frac = [], [], []
    blank = proc.tokenizer.pad_token_id
    for w in windows(audio, 30):
        x = proc(w, sampling_rate=SR, return_tensors="pt").input_values.to(device, torch.float16)
        probs = model(x).logits[0].float().softmax(-1)
        ids = probs.argmax(-1)
        texts.append(proc.decode(ids.cpu()))
        nonblank = ids != blank
        blank_frac.append(1 - nonblank.float().mean().item())
        if nonblank.any():
            maxp.append(probs.max(-1).values[nonblank].mean().item())
    return {"ctc_text": " ".join(t for t in texts if t).lower(),
            "ctc_conf_mean": float(np.mean(maxp)) if maxp else 0.0,
            "ctc_blank_frac": float(np.mean(blank_frac))}


MODELS = {
    "wavlm": ("microsoft/wavlm-large", load_wavlm, wavlm_clip),
    "ctc": ("facebook/wav2vec2-large-960h-lv60-self", load_ctc, ctc_clip),
}


def run(what, data_dir, out_dir, device, model_id=None, limit=0):
    default_id, loader, clip_fn = MODELS[what]
    proc, model = loader(model_id or default_id, device)
    clips = list_clips(Path(data_dir))[: limit or None]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if what == "ctc":
        with open(out_dir / "ctc.jsonl", "w", encoding="utf-8") as f:
            for i, (split, fn) in enumerate(clips):
                rec = ctc_clip(load_audio(Path(data_dir) / split / fn), proc, model, device)
                f.write(json.dumps({"split": split, "filename": fn, **rec}) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"[ctc] {i + 1}/{len(clips)}", flush=True)
        return

    means, stds = [], []
    for i, (split, fn) in enumerate(clips):
        m, s = clip_fn(load_audio(Path(data_dir) / split / fn), proc, model, device)
        means.append(m.astype(np.float16))
        stds.append(s.astype(np.float16))
        if (i + 1) % 50 == 0:
            print(f"[{what}] {i + 1}/{len(clips)}", flush=True)
    np.savez(out_dir / f"{what}.npz",
             split=np.array([c[0] for c in clips]), filename=np.array([c[1] for c in clips]),
             mean=np.stack(means), std=np.stack(stds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=MODELS)
    ap.add_argument("--data", default="data", help="folder with train.csv, test.csv, train/ and test/")
    ap.add_argument("--out", default="work/audio")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.what, a.data, a.out, a.device, a.model_id, a.limit)
