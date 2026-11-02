#!/usr/bin/env python3
"""Writes the golden files under tests/golden from the pipeline's own code, so the app's tests can
run without the pipeline. Needs fs_detailed on the path and the staged result files.

    python tests/tools/make_golden.py --pipeline ../fs_detailed --results <results folder> --stored stored.json

stored.json maps a text to its base64 float32 encoder vector (real bge-large output kept with the
model pages). The expected readings are computed with common.nn / common.nn_multi (torch) and the
expected word readings with common.bow, which is the same arithmetic the training and the
prediction scripts used.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--stored", required=True)
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "golden"))
args = ap.parse_args()
sys.path.insert(0, args.pipeline)
from common import bow, data, nn as heads, nn_multi as mh  # noqa: E402

R = Path(args.results)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
stored = json.loads(Path(args.stored).read_text(encoding="utf-8"))
texts = list(stored)
X = np.stack([np.frombuffer(base64.b64decode(stored[t]), "<f4") for t in texts]).astype(np.float32)

NETWORKS = {
    "fingpt_network": ("fd/train/nn/nn_model.json", False),
    "fingpt_network_sentfin_all": ("fd/rl/modelA_sentfin_mistakes/exact_all/nn_model.json", True),
    "fingpt_network_sentfin_mistakes": ("fd/rl/modelA_sentfin_mistakes/exact_wrong/nn_model.json", True),
    "fingpt_network_sentfin_grpo": ("fd/rl/modelA_sentfin_mistakes/grpo_all/nn_model.json", True),
    "dialogue_network": ("ed/train/nn/nn_model.json", True),
    "dialogue_network_fingpt": ("ed/rl/suite_standard_train/grpo_full/nn_model.json", True),
    "dialogue_network_fingpt_sentfin_mistakes": ("ed/rl/ed_fingpt_then_sentfin_mistakes/exact_wrong/nn_model.json", True),
    "dialogue_network_fingpt_sentfin_all": ("ed/rl/ed_fingpt_then_sentfin_mistakes/exact_all/nn_model.json", True),
    "dialogue_network_fingpt_sentfin_grpo": ("ed/rl/ed_fingpt_then_sentfin_mistakes/grpo_all/nn_model.json", True),
    "fingpt_linear": ("fd/train/linear/linear_model.json", True),
    "dialogue_linear": ("ed/train/linear/linear_model.json", True),
}
readings = {"texts": texts, "vectors": [stored[t] for t in texts], "models": {}}
for key, (rel, multi) in NETWORKS.items():
    mj = json.loads((R / rel).read_text(encoding="utf-8"))
    Z = heads.standardise_with(mj, X)
    out = {}
    if multi:
        if "config" not in mj:
            mj = dict(mj, config={"widths": [], "acts": []})
        model = mh.model_from_json(mj)
        temps = [float(t["temperature"]) for t in mj["tasks"]]
        for t, P in zip(mj["tasks"], mh.predict_proba(model, Z, temps)):
            out[t["name"]] = {"classes": t["classes"], "p": np.round(P, 7).tolist()}
    else:
        model = heads.model_from_json(mj)
        P = heads.predict_proba(model, Z, temperature=float(mj["temperature"]))
        out["sentiment"] = {"classes": mj["classes"], "p": np.round(P, 7).tolist()}
    readings["models"][key] = out
    print(key, {k: v["p"][0][:3] for k, v in out.items()})
(OUT / "readings.json").write_text(json.dumps(readings), encoding="utf-8")

WORD_TEXTS = [
    "Net sales for the first quarter grew 12% year-over-year, driven by stronger-than-expected demand in core enterprise segments",
    "Infosys shares plunge after weak guidance",
    "I can't wait to see my family this weekend, I'm so excited!",
    "Profits collapsed and the zzzunknownword board resigned",
    "qwertyuiop zxcvbnm",
    "12345 !!! $$$",
    "",
    "loss loss loss",
    "The company" + chr(0x2019) + "s outlook is terrific",
    "SHARES JUMPED AFTER THE BEAT",
    "it made me mad when i saw a dent in my car door last week",
    "profit of " + chr(0xE2) + chr(0x201A) + chr(0xAC) + "2.1bn beat forecasts",
]
words = {"texts": WORD_TEXTS, "cleaned": [data.clean_text(t) for t in WORD_TEXTS], "models": {}}
for key, d in (("words_fingpt", "fingpt"), ("words_sentfin", "sentfin"), ("words_dialogue", "empathetic")):
    mj = json.loads((R / "bow" / d / "bow_model.json").read_text(encoding="utf-8"))
    W, b = np.array(mj["coef"], np.float32), np.array(mj["intercept"], np.float32)
    index = {w: j for j, w in enumerate(mj["vocab"])}
    rows = []
    for t in words["cleaned"]:
        r = bow.read_text(t, index, W, b)
        rows.append({"p": r["p"], "words": [w["word"] for w in r["words"]], "unknown": [w["word"] for w in r["words"] if not w["known"]],
                     "per_word": {w["word"]: w["p"] for w in r["words"] if w["known"]}})
    words["models"][key] = {"classes": mj["class_names"], "rows": rows}
    print(key, [round(v, 4) for v in rows[0]["p"]][:3], rows[3]["unknown"])
(OUT / "words.json").write_text(json.dumps(words), encoding="utf-8")

CLEAN_CASES = ["  a \t b\n\n c  ", chr(0xFEFF) + "a" + chr(0x200B) + "b" + chr(0xA0) + "c", "yes_comma_ it was", "cafe" + chr(0x301),
               "Really?! $AAPL +5%...", "Monday" + chr(0xC2) + " ahead", "profit of " + chr(0xE2) + chr(0x201A) + chr(0xAC) + "2.1bn",
               (chr(0xE2) + chr(0x2026)) * 6, "The company" + chr(0x2019) + "s don't", "42", "", "   "]
cleaning = [{"raw": c, "clean": data.clean_text(c), "tokens": bow.tokens(data.clean_text(c))} for c in CLEAN_CASES]
(OUT / "cleaning.json").write_text(json.dumps(cleaning), encoding="utf-8")
print("golden files written to", OUT)
