"""Transcribe every clip with Whisper large-v3 (faster-whisper) and save one JSON per clip.

Keeps what grammar scoring needs and a plain transcript throws away: word timestamps and
confidences (for pauses and fluency), segment log-probabilities, and the detected language.

Run twice, once per transcript style:

    python extraction/transcribe.py --tag disfluent --prompt disfluent    # word-for-word
    python extraction/transcribe.py --tag clean --prompt none             # Whisper's default

On a machine with several GPUs, start one process per GPU with --shard i --nshards N --device_index i;
each process writes its own subset of the files.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch  # noqa: F401  (loads the CUDA libraries that ctranslate2 needs on Windows)
import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel

# A disfluent prompt nudges Whisper to keep fillers, repetitions and false starts
# instead of silently cleaning them up (Whisper otherwise writes the "intended" text).
PROMPTS = {
    "none": None,
    "disfluent": "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. "
                 "I- I was go to the, uh, the market and, um, I buyed some.",
}


def list_clips(data_dir):
    rows = []
    for split in ("train", "test"):
        df = pd.read_csv(data_dir / f"{split}.csv")
        rows += [(split, f) for f in df["filename"]]
    return rows


def load_audio(path):
    """All clips are 16 kHz mono PCM16, so soundfile reads them directly (no resampling)."""
    audio, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000, f"{path}: unexpected sample rate {sr}"
    return audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)


def transcribe_one(model, path, prompt):
    audio = load_audio(path)

    # Language ID on the first 30 s and, separately, on the rest (a speaker may switch).
    lang, lang_prob, all_probs = model.detect_language(audio)
    top_langs = sorted(all_probs, key=lambda x: -x[1])[:5]
    top_langs2 = None
    if len(audio) > 35 * 16000:
        _, _, all_probs2 = model.detect_language(audio[30 * 16000:])
        top_langs2 = sorted(all_probs2, key=lambda x: -x[1])[:5]

    segments, info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=5,
        # Temperature fallback re-decodes a segment when it looks like a repetition
        # loop (high compression ratio) or has very low confidence.
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        condition_on_previous_text=False,  # stops one bad segment from poisoning the next
        initial_prompt=prompt,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segs, words = [], []
    for s in segments:
        segs.append({
            "start": s.start, "end": s.end, "text": s.text,
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "compression_ratio": s.compression_ratio, "temperature": s.temperature,
        })
        for w in s.words or []:
            words.append({"w": w.word, "s": w.start, "e": w.end, "p": w.probability})

    return {
        "duration": len(audio) / 16000,
        "lang": lang, "lang_prob": lang_prob, "top_langs": top_langs, "top_langs2": top_langs2,
        "text": " ".join(s["text"].strip() for s in segs),
        "segments": segs, "words": words,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with train.csv, test.csv, train/ and test/")
    ap.add_argument("--out", default="work/asr")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--tag", required=True, help="output subfolder under --out")
    ap.add_argument("--prompt", default="none", choices=PROMPTS)
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument("--device_index", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (for timing)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = list_clips(data_dir)
    if args.limit:
        clips = clips[: args.limit]
    clips = clips[args.shard::args.nshards]

    model = WhisperModel(args.model, device="cuda", device_index=args.device_index,
                         compute_type=args.compute_type)

    t0, done = time.time(), 0
    for i, (split, fname) in enumerate(clips):
        out = out_dir / f"{split}__{fname}.json"
        if out.exists():
            continue
        try:
            res = transcribe_one(model, data_dir / split / fname, PROMPTS[args.prompt])
        except Exception as e:  # record the failure instead of silently writing an empty transcript
            res = {"error": repr(e)}
            print(f"ERROR {split}/{fname}: {e}", file=sys.stderr)
        res.update({"split": split, "filename": fname, "model": args.model, "prompt": args.prompt})
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out)
        done += 1
        if done % 10 == 0 or i == len(clips) - 1:
            el = time.time() - t0
            print(f"[{args.tag} shard {args.shard}] {i + 1}/{len(clips)}  {el / done:.1f}s/clip  "
                  f"eta {(len(clips) - i - 1) * el / done / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
