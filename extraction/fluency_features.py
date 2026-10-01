"""Fluency and recognition-confidence features from the word timestamps of the word-for-word transcript.

Every feature is a rate or a ratio, never a raw count: test clips are shorter (median 45 s) than
training clips (median 60 s), so counts would shift between the two sets.

    python extraction/fluency_features.py
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "umm", "uhm", "eh", "mhm"}
PAUSE_THRESHOLDS = (0.25, 0.5, 1.0)


def norm_word(w):
    return re.sub(r"[^a-z']", "", w.lower())


def features_from_asr(rec):
    words = [w for w in rec.get("words", []) if norm_word(w["w"])]
    segs = rec.get("segments", [])
    dur = rec.get("duration", np.nan)
    f = {"lang_prob_en": dict(rec.get("top_langs", [])).get("en", 0.0),
         "lang_prob_en_2nd": dict(rec.get("top_langs2") or []).get("en", np.nan),
         "lang_is_en": float(rec.get("lang") == "en")}
    n = len(words)
    f["n_words_per_min"] = n / dur * 60 if dur else np.nan
    if n < 3:
        f["too_few_words"] = 1.0
        return f
    f["too_few_words"] = 0.0

    toks = [norm_word(w["w"]) for w in words]
    starts = np.array([w["s"] for w in words])
    ends = np.array([w["e"] for w in words])
    probs = np.array([w["p"] for w in words])
    span = ends[-1] - starts[0]
    gaps = np.clip(starts[1:] - ends[:-1], 0, None)

    f["speech_span_ratio"] = span / dur
    f["lead_silence"] = starts[0] / dur
    f["speech_rate_wps"] = n / span
    for t in PAUSE_THRESHOLDS:
        p = gaps > t
        f[f"pauses_{t}_per_min"] = p.sum() / span * 60
    long_p = gaps > PAUSE_THRESHOLDS[0]
    f["mean_pause_s"] = gaps[long_p].mean() if long_p.any() else 0.0
    f["pause_time_ratio"] = gaps[long_p].sum() / span
    # Mean length of run: words spoken between pauses longer than 0.25 s.
    runs = np.diff(np.flatnonzero(np.r_[True, long_p, True]))
    f["mean_len_run"] = runs.mean()
    f["articulation_rate"] = n / max(span - gaps[long_p].sum(), 1e-3)

    f["filler_per_100w"] = 100 * sum(t in FILLERS for t in toks) / n
    f["repeat_per_100w"] = 100 * sum(a == b for a, b in zip(toks, toks[1:])) / n
    bigrams = list(zip(toks, toks[1:]))
    f["bigram_repeat_per_100w"] = 100 * sum(b1 == b2 for b1, b2 in zip(bigrams, bigrams[2:])) / n
    content = [t for t in toks if t not in FILLERS]
    f["type_token_ratio_50"] = np.mean([len(set(content[i:i + 50])) / 50
                                        for i in range(0, max(len(content) - 49, 1), 10)]) \
        if len(content) >= 50 else len(set(content)) / max(len(content), 1)

    f["word_prob_mean"] = probs.mean()
    f["word_prob_p10"] = np.percentile(probs, 10)
    f["word_prob_lt05"] = (probs < 0.5).mean()
    if segs:
        f["seg_logprob_mean"] = np.mean([s["avg_logprob"] for s in segs])
        f["seg_logprob_min"] = np.min([s["avg_logprob"] for s in segs])
        f["no_speech_mean"] = np.mean([s["no_speech_prob"] for s in segs])
        f["compression_mean"] = np.mean([s["compression_ratio"] for s in segs])
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="work/asr/disfluent", help="folder with the word-for-word transcripts")
    ap.add_argument("--out", default="work/text/fluency.parquet")
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.asr).glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        feats = features_from_asr(rec) if "error" not in rec else {}
        rows.append({"split": rec["split"], "filename": rec["filename"], **feats})
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print("saved", a.out, df.shape)
