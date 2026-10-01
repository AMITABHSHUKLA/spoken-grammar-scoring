"""Voxtral-Mini-3B (an audio language model) listens to each clip together with a grammar question.
Saves the mean hidden state over the audio tokens for the lower-middle layers of its language model.

    python extraction/voxtral_features.py

Outputs in --out: voxtral_amean.npy [clips, layers, hidden] (float16), voxtral_meta.json (which layers), keys.csv.
Tested with transformers 5.18.0 and mistral-common[audio] 1.12.0.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR = 16000
CHUNK_TOKENS = 375      # Voxtral audio tokens per 30 s chunk
TOKENS_PER_SEC = 12.5
PROMPT = ("Listen to this spoken answer from an English test. "
          "How accurate and complex is the speaker's grammar?")


def clips(data):
    rows = []
    for split in ("train", "test"):
        rows += [(split, f) for f in pd.read_csv(Path(data) / f"{split}.csv").filename]
    return rows


def load_fp16(cls, name, **kw):
    """from_pretrained in fp16; the keyword is `dtype` in new transformers and `torch_dtype` in older ones."""
    try:
        return cls.from_pretrained(name, dtype=torch.float16, **kw)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=torch.float16, **kw)


def load_audio(path):
    y, sr = sf.read(str(path), dtype="float32")
    assert sr == SR
    return y if y.ndim == 1 else y.mean(1)


@torch.no_grad()
def run_voxtral(args, rows, out):
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    from transformers.processing_utils import ProcessorMixin
    # Some transformers versions crash while LOGGING the processor (its tokenizer has no working repr).
    ProcessorMixin.__repr__ = lambda self: f"<{type(self).__name__}>"
    proc = AutoProcessor.from_pretrained(args.processor or args.model)
    model = load_fp16(VoxtralForConditionalGeneration, args.model).to(args.device).eval()
    audio_id = model.config.audio_token_id
    L = model.config.text_config.num_hidden_layers
    layers = list(range(int(0.2 * L), int(0.6 * L) + 1))          # lower-middle layers of the language model
    json.dump({"layers": layers, "model": args.model}, open(out / "voxtral_meta.json", "w"))
    amean, bad = [], 0
    tmp = out / "_voxtral_clip.wav"
    t0 = time.time()
    for i, (split, fn) in enumerate(rows):
        # Voxtral pads audio to 30 s chunks (375 audio tokens each, 12.5 per second). A 60.1 s clip would
        # get a third chunk with 0.1 s of audio and 29.9 s of padding, so leftovers under 2 s are trimmed.
        y = load_audio(Path(args.data) / split / fn)
        dur = len(y) / SR
        if dur > 30 and dur % 30 < 2.0:
            y = y[: int((dur - dur % 30) * SR)]
            dur = len(y) / SR
        sf.write(tmp, y, SR)
        conv = [{"role": "user", "content": [{"type": "audio", "path": str(tmp)}, {"type": "text", "text": PROMPT}]}]
        inputs = proc.apply_chat_template(conv).to(args.device, dtype=torch.float16)
        hs = model(**inputs, output_hidden_states=True).hidden_states   # (layers+1) x [1, T, H]
        H = torch.stack([hs[l][0].float() for l in layers])              # [len(layers), T, H]
        if not bool(torch.isfinite(H).all()):  # fp16 overflow: keep going, mark the clip as missing
            bad += 1
            H = torch.full_like(H, float("nan"))
        # Average only over audio tokens that cover real audio, not the padding of the last chunk.
        pos = torch.nonzero(inputs["input_ids"][0] == audio_id).squeeze(-1)
        j = torch.arange(len(pos), device=pos.device)
        chunk, within = j // CHUNK_TOKENS, j % CHUNK_TOKENS
        real_tokens = torch.ceil(torch.clamp(dur - 30.0 * chunk, 0, 30.0) * TOKENS_PER_SEC)
        keep = pos[within < real_tokens]
        amean.append(H[:, keep].mean(1).half().cpu().numpy())
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[voxtral] {i + 1}/{len(rows)}  {(time.time() - t0) / (i + 1):.2f}s/clip  "
                  f"duration {dur:.1f}s: audio tokens {len(pos)}, real {len(keep)}  non-finite so far {bad}", flush=True)
    np.save(out / "voxtral_amean.npy", np.stack(amean))
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with train.csv, test.csv, train/ and test/")
    ap.add_argument("--out", default="work/voxtral")
    ap.add_argument("--model", default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--processor", default=None, help="processor folder, if different from --model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = clips(a.data)[: a.limit or None]
    pd.DataFrame(rows, columns=["split", "filename"]).to_csv(out / "keys.csv", index=False)
    run_voxtral(a, rows, out)
    print("done", flush=True)
