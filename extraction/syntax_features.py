"""Syntax and vocabulary features from spaCy parses, aimed at the rubric: incomplete sentences,
clause complexity, subject-verb agreement errors, tense and verb-form variety, and word rarity.
All are rates or ratios.

Computed on the word-for-word transcript (prefix syn_; fillers removed so they do not break the parse)
and on the LLM's minimally corrected version (prefix sync_, "what the speaker meant").

    python -m spacy download en_core_web_sm
    python extraction/syntax_features.py
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from wordfreq import zipf_frequency

FILLERS = r"\b(um+|uh+|erm|er|ah|hmm+|mm+|uhm|eh|mhm)\b[,.]?\s*"
SUBORD = {"advcl", "ccomp", "acl", "relcl", "csubj", "csubjpass"}
SUBJ = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
POS_KEEP = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "CCONJ", "SCONJ", "AUX")
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
SING_PRON, PLUR_PRON = {"he", "she", "it", "this", "that"}, {"i", "you", "we", "they", "these", "those"}


def clean(text):
    text = re.sub(FILLERS, "", text, flags=re.I)
    text = re.sub(r"\b(\w+)(?:[,\s-]+\1\b)+", r"\1", text, flags=re.I)  # collapse "the, the, the"
    return re.sub(r"\s+", " ", text).strip()


def depth(tok):
    d = 0
    while tok.head.i != tok.i:
        tok, d = tok.head, d + 1
    return d


def finite_verb(head):
    """The finite verb that should agree with a subject attached to `head`."""
    if "Fin" in head.morph.get("VerbForm"):
        return head
    for c in head.children:
        if c.dep_ in ("aux", "auxpass") and "Fin" in c.morph.get("VerbForm"):
            return c
    return None


def agreement_error(subj):
    v = finite_verb(subj.head)
    if v is None or v.tag_ not in ("VBZ", "VBP"):
        return False
    low = subj.lower_
    if subj.pos_ == "PRON":
        if low in SING_PRON:
            return v.tag_ == "VBP"
        if low in PLUR_PRON:
            return v.tag_ == "VBZ" and not (low == "i" and v.lower_ in ("am", "'m"))
        return False
    if subj.tag_ in ("NNS", "NNPS"):
        return v.tag_ == "VBZ"
    if subj.tag_ in ("NN", "NNP") and not any(c.dep_ == "cc" for c in subj.children):
        return v.tag_ == "VBP"
    return False


def doc_features(doc):
    words = [t for t in doc if t.is_alpha]
    n = max(len(words), 1)
    per100 = lambda k: 100 * k / n
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    f = {}
    lens = [sum(t.is_alpha for t in s) for s in sents] or [0]
    f["sents_per_100w"] = per100(len(sents))
    f["sent_len_mean"], f["sent_len_std"], f["sent_len_max"] = np.mean(lens), np.std(lens), np.max(lens)

    no_verb = no_subj = 0
    clauses, depths = [], []
    for s in sents:
        verbs = [t for t in s if t.pos_ in ("VERB", "AUX")]
        no_verb += not verbs
        no_subj += bool(verbs) and not any(t.dep_ in SUBJ for t in s)
        clauses.append(sum("Fin" in t.morph.get("VerbForm") for t in s))
        depths.append(max(depth(t) for t in s))
    k = max(len(sents), 1)
    f["frac_sent_no_verb"], f["frac_sent_no_subj"] = no_verb / k, no_subj / k
    f["clauses_per_sent"] = np.mean(clauses) if clauses else 0.0
    f["tree_depth_mean"] = np.mean(depths) if depths else 0.0
    f["tree_depth_max"] = np.max(depths) if depths else 0.0
    f["dep_dist_mean"] = np.mean([abs(t.head.i - t.i) for t in doc if t.is_alpha]) if words else 0.0

    deps = pd.Series([t.dep_ for t in doc])
    f["subord_per_100w"] = per100(deps.isin(SUBORD).sum())
    for d in ("relcl", "advcl", "ccomp", "xcomp", "mark", "cc", "prep", "nsubjpass"):
        f[f"{d}_per_100w"] = per100((deps == d).sum())
    tags = pd.Series([t.tag_ for t in doc])
    f["modal_per_100w"] = per100((tags == "MD").sum())

    fin = [t for t in doc if "Fin" in t.morph.get("VerbForm")]
    f["past_ratio"] = np.mean(["Past" in t.morph.get("Tense") for t in fin]) if fin else 0.0
    verb_forms = {(t.tag_, t.dep_ in ("aux", "auxpass")) for t in doc if t.pos_ in ("VERB", "AUX")}
    f["verb_form_variety"] = len(verb_forms)
    f["perfect_per_100w"] = per100(sum(t.tag_ == "VBN" and any(c.lemma_ == "have" and c.dep_ == "aux" for c in t.children) for t in doc))
    f["progressive_per_100w"] = per100(sum(t.tag_ == "VBG" and any(c.lemma_ == "be" and c.dep_ == "aux" for c in t.children) for t in doc))

    f["agreement_err_per_100w"] = per100(sum(agreement_error(t) for t in doc if t.dep_ in ("nsubj", "nsubjpass")))
    f["a_plural_err_per_100w"] = per100(sum(t.lower_ in ("a", "an") and t.dep_ == "det" and t.head.tag_ == "NNS" for t in doc))

    pos = pd.Series([t.pos_ for t in words])
    for p in POS_KEEP:
        f[f"pos_{p.lower()}_ratio"] = (pos == p).mean() if len(pos) else 0.0
    content = [t for t in words if t.pos_ in CONTENT]
    zipf = np.array([zipf_frequency(t.lemma_.lower(), "en") for t in content]) if content else np.array([7.0])
    f["lex_density"] = len(content) / n
    f["word_zipf_mean"] = zipf.mean()
    f["rare_word_ratio"] = (zipf < 4.0).mean()
    f["word_len_mean"] = np.mean([len(t.text) for t in words]) if words else 0.0
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="work/asr/disfluent", help="folder with the word-for-word transcripts")
    ap.add_argument("--llm", default="work/text/llm_features.parquet", help="for the corrected transcripts")
    ap.add_argument("--out", default="work/text/syntax.parquet")
    a = ap.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    rows = []
    for p in sorted(Path(a.asr).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"split": r["split"], "filename": r["filename"], "text": clean(r.get("text", ""))})
    df = pd.DataFrame(rows)
    llm = pd.read_parquet(a.llm)
    df = df.merge(llm[["split", "filename", "corrected_text"]], on=["split", "filename"], how="left")

    out = df[["split", "filename"]].copy()
    for col, prefix in (("text", "syn_"), ("corrected_text", "sync_")):
        feats = [doc_features(d) for d in nlp.pipe(df[col].fillna("").map(clean), batch_size=64)]
        out = pd.concat([out, pd.DataFrame(feats).add_prefix(prefix)], axis=1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print("saved", a.out, out.shape)
