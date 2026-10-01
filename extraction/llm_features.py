"""LLM features from the transcripts: a rubric judge and a minimal grammar correction.

Judge: the model reads the official rubric and a transcript and answers with one digit. From the
log-probabilities of that first answer token we take the probability of each of 1-5 and their weighted
average, which gives a continuous score instead of one rounded answer.

Correction: the model rewrites the transcript, fixing only the grammar. The rate of word edits between
the transcript and its correction measures the grammatical errors directly.

Any OpenAI-compatible chat endpoint that returns token log-probabilities works. I used the open-weights
model Qwen3.5-9B (4-bit AWQ) served with vLLM. Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in the
environment or in a .env file. Every response is cached on disk, so a re-run costs nothing.

    python extraction/llm_features.py
"""
import argparse
import difflib
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RUBRIC = """Grammar score rubric:
1 - Struggles with proper sentence structure and syntax; limited control over simple grammatical structures and memorized sentence patterns.
2 - Limited understanding of sentence structure and syntax. Uses simple structures but consistently makes basic sentence-structure and grammatical mistakes; may leave sentences incomplete.
3 - Decent grasp of sentence structure but errors in grammatical structure, or decent grasp of grammatical structure but errors in sentence syntax and structure.
4 - Strong understanding of sentence structure and syntax; consistently good control of grammar. Occasional minor errors that do not cause misunderstanding; can correct most of them.
5 - High grammatical accuracy and adept control of complex grammar. Uses grammar accurately and effectively, seldom making noticeable mistakes; handles complex structures well and self-corrects when necessary."""

JUDGE_SYSTEM = f"""You are an expert examiner scoring the GRAMMAR of spontaneous spoken English.
The input is an automatic speech-recognition transcript of a 45-60 second spoken answer.
Ignore punctuation, capitalisation and likely transcription errors on names. Fillers
(um, uh) and self-corrections are normal in speech; judge sentence structure, syntax,
and grammatical accuracy and complexity, as the rubric describes.

{RUBRIC}

Reply with a single digit from 1 to 5 and nothing else."""

GEC_SYSTEM = """You correct grammatical errors in transcripts of spoken English.
Rewrite the transcript with the MINIMUM changes needed to make each sentence
grammatical: fix tense, agreement, articles, prepositions, word order, missing or
extra words, and incomplete sentences. Remove fillers (um, uh) and stutters.
Do not paraphrase, improve style, or change vocabulary that is already correct.
Output only the corrected transcript."""

FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "umm", "uhm", "eh", "mhm"}

_client = None
CACHE = Path("work/llm_cache")


def chat(messages, model, max_tokens, **kwargs):
    """Chat completion at temperature 0 with a disk cache keyed by the full request."""
    global _client
    req = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},   # no reasoning preamble (Qwen on vLLM)
           **kwargs}
    path = CACHE / (hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:32] + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    if _client is None:
        _client = OpenAI(api_key=os.environ["LLM_API_KEY"].strip(), base_url=os.environ["LLM_BASE_URL"].strip(), timeout=120)
    for attempt in range(4):  # transient network or server errors
        try:
            resp = _client.chat.completions.create(**req).model_dump()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    path.write_text(json.dumps({"request": req, "response": resp}, ensure_ascii=False), encoding="utf-8")
    return resp


def judge(text, model):
    """Returns (expected score, {digit: probability}) from the log-probabilities of the first answer token."""
    r = chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "Transcript:\n" + text}],
             model, max_tokens=5, logprobs=True, top_logprobs=10)
    probs = {}
    for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        tok = t["token"].strip()
        if tok in {"1", "2", "3", "4", "5"}:
            probs[int(tok)] = probs.get(int(tok), 0.0) + math.exp(t["logprob"])
    z = sum(probs.values())
    if z == 0:
        return np.nan, {}
    probs = {k: v / z for k, v in probs.items()}
    return sum(k * p for k, p in probs.items()), probs


def correct(text, model):
    r = chat([{"role": "system", "content": GEC_SYSTEM}, {"role": "user", "content": text}],
             model, max_tokens=int(len(text.split()) * 2.5) + 50)
    return r["choices"][0]["message"]["content"].strip()


def _toks(s):
    # Fillers are dropped so that their removal by the corrector is not counted as a grammar edit.
    return [t for t in re.findall(r"[a-z']+", s.lower()) if t not in FILLERS]


def edit_features(src, corrected):
    """Word-level edits between the transcript and its correction, per 100 words."""
    a, b = _toks(src), _toks(corrected)
    n = max(len(a), 1)
    ops = {"replace": 0, "delete": 0, "insert": 0}
    changed_words = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            ops[tag] += 1
            changed_words += max(i2 - i1, j2 - j1)
    return {
        "gec_edits_per_100w": 100 * sum(ops.values()) / n,
        "gec_changed_words_pct": 100 * changed_words / n,
        "gec_replace_per_100w": 100 * ops["replace"] / n,
        "gec_delete_per_100w": 100 * ops["delete"] / n,
        "gec_insert_per_100w": 100 * ops["insert"] / n,
    }


def run_many(fn, texts, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, texts))


def load_texts(folder):
    rows = []
    for p in sorted(Path(folder).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"split": r["split"], "filename": r["filename"], "text": r.get("text", "")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="work/asr", help="folder with the disfluent/ and clean/ transcripts")
    ap.add_argument("--out", default="work/text/llm_features.parquet")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "").strip())
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    CACHE = Path(a.cache)
    CACHE.mkdir(parents=True, exist_ok=True)

    df = load_texts(Path(a.asr) / "disfluent").merge(load_texts(Path(a.asr) / "clean"), on=["split", "filename"],
                                                     suffixes=("_disfluent", "_clean"))
    print(f"model {a.model}, {len(df)} clips", flush=True)
    out = df[["split", "filename"]].copy()
    for tag in ("disfluent", "clean"):       # rubric score on the word-for-word and on the default transcript
        res = run_many(lambda t: judge(t, a.model), list(df[f"text_{tag}"]), a.workers)
        out[f"judge_{tag}"] = [r[0] for r in res]
        for k in range(1, 6):
            out[f"judge_{tag}_p{k}"] = [r[1].get(k, 0.0) for r in res]
    corrected = run_many(lambda t: correct(t, a.model), list(df.text_disfluent), a.workers)
    edits = pd.DataFrame([edit_features(src, cor) for src, cor in zip(df.text_disfluent, corrected)])
    out = pd.concat([out, edits], axis=1)
    out["corrected_text"] = corrected
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print("saved", a.out, out.shape)
