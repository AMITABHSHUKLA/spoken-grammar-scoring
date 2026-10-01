"""Assemble the small `features/` folder that the notebook reads.

Inputs are the outputs of the other extraction scripts (by default in `work/`, where those scripts
write them). Rows are ordered as train.csv followed by test.csv.

    python extraction/build_features.py

Outputs
    features/meta.csv               split, filename, duration (s), ctc_blank_frac (share of frames with no speech sound)
    features/voice.npy              WavLM layers 3 and 6, mean and std over time (voice identity; used to group speakers)
    features/wavlm.npy              WavLM layer 20, mean over time (how the speech sounds)
    features/voxtral.npy            Voxtral language-model layers 12-14, mean over the audio tokens
    features/text_features.parquet  LLM rubric scores, grammar-edit rates, fluency and syntax features
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def order_of(keys, wanted):
    """Row positions of `wanted` (split, filename) pairs inside a keys table."""
    pos = {k: i for i, k in enumerate(zip(keys["split"], keys["filename"]))}
    return np.array([pos[k] for k in wanted])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with train.csv and test.csv")
    ap.add_argument("--work", default="work", help="folder with the extraction outputs")
    ap.add_argument("--out", default="features")
    ap.add_argument("--asr", default=None, help="word-for-word transcripts (default: <work>/asr/disfluent)")
    ap.add_argument("--wavlm", default=None, help="default: <work>/audio/wavlm.npz")
    ap.add_argument("--ctc", default=None, help="default: <work>/audio/ctc.jsonl")
    ap.add_argument("--voxtral", default=None, help="folder with voxtral_amean.npy, voxtral_meta.json, keys.csv")
    ap.add_argument("--llm", default=None, help="default: <work>/text/llm_features.parquet")
    ap.add_argument("--fluency", default=None, help="default: <work>/text/fluency.parquet")
    ap.add_argument("--syntax", default=None, help="default: <work>/text/syntax.parquet")
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(Path(a.data) / "train.csv")
    test = pd.read_csv(Path(a.data) / "test.csv")
    rows = [("train", f) for f in train.filename] + [("test", f) for f in test.filename]
    meta = pd.DataFrame(rows, columns=["split", "filename"])

    # Duration from the transcripts' metadata; share of CTC "blank" frames from the wav2vec2 pass.
    asr = Path(a.asr or work / "asr" / "disfluent")
    meta["duration"] = [json.loads((asr / f"{s}__{f}.json").read_text(encoding="utf-8"))["duration"] for s, f in rows]
    ctc = pd.read_json(a.ctc or work / "audio" / "ctc.jsonl", lines=True)
    meta["ctc_blank_frac"] = ctc.ctc_blank_frac.values[order_of(ctc, rows)]
    meta.to_csv(out / "meta.csv", index=False)

    z = np.load(a.wavlm or work / "audio" / "wavlm.npz")
    o = order_of({"split": z["split"], "filename": z["filename"]}, rows)
    mean, std = z["mean"][o], z["std"][o]
    np.save(out / "voice.npy", np.hstack([mean[:, 3], std[:, 3], mean[:, 6], std[:, 6]]).astype(np.float16))
    np.save(out / "wavlm.npy", mean[:, 20].astype(np.float32))

    vdir = Path(a.voxtral or work / "voxtral")
    layers = json.load(open(vdir / "voxtral_meta.json"))["layers"]
    vox = np.load(vdir / "voxtral_amean.npy")[order_of(pd.read_csv(vdir / "keys.csv"), rows)]
    pick = [layers.index(k) for k in (12, 13, 14)]
    np.save(out / "voxtral.npy", vox[:, pick].astype(np.float32).mean(1))

    llm = pd.read_parquet(a.llm or work / "text" / "llm_features.parquet").drop(columns=["corrected_text"])
    text = (llm.merge(pd.read_parquet(a.fluency or work / "text" / "fluency.parquet"), on=["split", "filename"])
               .merge(pd.read_parquet(a.syntax or work / "text" / "syntax.parquet"), on=["split", "filename"]))
    text = meta[["split", "filename"]].merge(text, on=["split", "filename"], how="left")
    text.to_parquet(out / "text_features.parquet", index=False)
    print(f"{len(meta)} clips | voice {np.load(out / 'voice.npy').shape} | wavlm {np.load(out / 'wavlm.npy').shape} | "
          f"voxtral {np.load(out / 'voxtral.npy').shape} | text features {text.shape[1] - 2}")


if __name__ == "__main__":
    main()
