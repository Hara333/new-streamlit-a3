"""Build models/ for app.py from the pipeline's result files.

    python build_models.py --results <folder laid out like $COS_ROOT/results> [--logs <folder of job logs>] --out models

Each model becomes one .npz file (float32 weights, loaded with allow_pickle=False) and one entry in
models/catalog.json: its name, method, readings, the numbers the details panel shows and the curve
it draws. Nothing is retrained; the weights are the ones the pipeline exported.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

SHADES_TO_POLARITY = {"strong negative": "negative", "negative": "negative", "moderately negative": "negative", "mildly negative": "negative",
                      "neutral": "neutral", "mildly positive": "positive", "moderately positive": "positive", "positive": "positive",
                      "strong positive": "positive"}
ENCODER = "bge-large-en-v1.5, frozen: the [CLS] vector and the mean over the tokens of the last layer, 2,048 values"

MODELS = [
    {"key": "fingpt_network", "name": "FinGPT Network", "tag": "deep learning", "kind": "network", "base": "fd/train/nn"},
    {"key": "fingpt_network_sentfin_all", "name": "FinGPT Network + SEntFiN, all rows", "tag": "deep learning, reinforced: closed form",
     "kind": "network", "base": "fd/train/nn", "stages": ["fd/rl/modelA_sentfin_mistakes/exact_all"]},
    {"key": "fingpt_network_sentfin_mistakes", "name": "FinGPT Network + SEntFiN, mistakes only", "tag": "deep learning, reinforced: closed form",
     "kind": "network", "base": "fd/train/nn", "stages": ["fd/rl/modelA_sentfin_mistakes/exact_wrong"]},
    {"key": "fingpt_network_sentfin_grpo", "name": "FinGPT Network + SEntFiN, GRPO", "tag": "deep learning, reinforced: GRPO",
     "kind": "network", "base": "fd/train/nn", "stages": ["fd/rl/modelA_sentfin_mistakes/grpo_all"]},
    {"key": "dialogue_network", "name": "Dialogue Network", "tag": "deep learning", "kind": "network", "base": "ed/train/nn"},
    {"key": "dialogue_network_fingpt", "name": "Dialogue Network + FinGPT", "tag": "deep learning, reinforced: GRPO",
     "kind": "network", "base": "ed/train/nn", "stages": ["ed/rl/suite_standard_train/grpo_full"]},
    {"key": "dialogue_network_fingpt_sentfin_mistakes", "name": "Dialogue Network + FinGPT + SEntFiN, mistakes only",
     "tag": "deep learning, reinforced twice: GRPO, then closed form", "kind": "network", "base": "ed/train/nn",
     "stages": ["ed/rl/suite_standard_train/grpo_full", "ed/rl/ed_fingpt_then_sentfin_mistakes/exact_wrong"]},
    {"key": "dialogue_network_fingpt_sentfin_all", "name": "Dialogue Network + FinGPT + SEntFiN, all rows",
     "tag": "deep learning, reinforced twice: GRPO, then closed form", "kind": "network", "base": "ed/train/nn",
     "stages": ["ed/rl/suite_standard_train/grpo_full", "ed/rl/ed_fingpt_then_sentfin_mistakes/exact_all"]},
    {"key": "dialogue_network_fingpt_sentfin_grpo", "name": "Dialogue Network + FinGPT + SEntFiN, GRPO",
     "tag": "deep learning, reinforced twice: GRPO, then GRPO", "kind": "network", "base": "ed/train/nn",
     "stages": ["ed/rl/suite_standard_train/grpo_full", "ed/rl/ed_fingpt_then_sentfin_mistakes/grpo_all"]},
    {"key": "fingpt_linear", "name": "FinGPT Linear", "tag": "linear", "kind": "linear", "src": "fd"},
    {"key": "dialogue_linear", "name": "Dialogue Linear", "tag": "linear", "kind": "linear", "src": "ed"},
    {"key": "words_fingpt", "name": "Word Logit, FinGPT", "tag": "MNLM", "kind": "words", "src": "bow/fingpt"},
    {"key": "words_sentfin", "name": "Word Logit, SEntFiN", "tag": "MNLM", "kind": "words", "src": "bow/sentfin"},
    {"key": "words_dialogue", "name": "Word Logit, Dialogue", "tag": "MNLM", "kind": "words", "src": "bow/empathetic"},
]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(v):
    return f"{100 * v:.1f}%"


def f3(v):
    return f"{v:.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def job_report(logs, prefix):
    """Wall time and peak memory from the scheduler's report at the end of a job log."""
    if not logs:
        return None
    for f in sorted(Path(logs).glob(prefix + "*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace")
        mem = re.search(r"Memory \(RAM\).*?\(([\d.]+ \w+) peak", text)
        tim = re.search(r"Time .*?\((\d+)-(\d+):(\d+):(\d+) /", text)
        if mem and tim:
            d, h, m, s = map(int, tim.groups())
            return {"memory": mem.group(1), "minutes": round(d * 1440 + h * 60 + m + s / 60, 1)}
    return None


# ----- the networks

def network_arrays(mj):
    """Hidden layers and one output layer per reading, from either export layout. The adapted
    FinGPT Network exports still carry the starting network under "layers"; their adapted weights
    are under "trunk" and "tasks", so those are read first."""
    if mj.get("trunk") is not None and mj.get("tasks") and "out" in mj["tasks"][0]:
        trunk = [(np.array(l["W"], np.float32), np.array(l["b"], np.float32), l["act"]) for l in mj["trunk"]]
        outs = [(t["name"], t["classes"], float(t["temperature"]), np.array(t["out"]["W"], np.float32), np.array(t["out"]["b"], np.float32)) for t in mj["tasks"]]
    else:
        layers = mj["layers"]
        trunk = [(np.array(l["W"], np.float32), np.array(l["b"], np.float32), l["act"]) for l in layers[:-1]]
        last = layers[-1]
        outs = [("sentiment", mj["classes"], float(mj["temperature"]), np.array(last["W"], np.float32), np.array(last["b"], np.float32))]
    return np.array(mj["mean"], np.float32), np.array(mj["sd"], np.float32), trunk, outs


def save_network(out, key, mj):
    mean, sd, trunk, outs = network_arrays(mj)
    arrays = {"mean": mean, "sd": sd}
    for i, (W, b, _) in enumerate(trunk):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    for name, _, _, W, b in outs:
        arrays[f"out_{name}_W"], arrays[f"out_{name}_b"] = W, b
    np.savez_compressed(out / f"{key}.npz", **arrays)
    params = sum(W.size + b.size for W, b, _ in trunk) + sum(W.size + b.size for _, _, _, W, b in outs)
    return {"acts": [a for _, _, a in trunk], "widths": [int(W.shape[0]) for W, _, _ in trunk],
            "readings": [{"name": n, "classes": c, "temperature": T} for n, c, T, _, _ in outs], "params": int(params)}


def sentfin_zero_shot(results, cfg):
    if cfg == "fd":
        for r in csv.DictReader(open(results / "fd/sentfin/prediction_metrics.csv", encoding="utf-8")):
            if r["subset"] == "all" and r["correction"] == "none":
                return float(r["accuracy"]), float(r["macro_f1"]), int(r["rows"])
    s = load(results / "ed/sentfin/prediction_summary.json")["readings"]["sentiment"]
    return s["accuracy"]["none"], s["macro_f1"]["none"], s["labelled_rows"]


def base_details(results, logs, cfg, info):
    mj = load(results / f"{cfg}/train/nn/nn_model.json")
    sel = load(results / f"{cfg}/train/nn/nn_selected.json")
    c, o = mj["config"], mj["optimiser"]
    sgd = mj["sgd"]
    acc, f1, n_sf = sentfin_zero_shot(results, cfg)
    lin = load(results / f"{cfg}/train/linear/linear_summary.json")["readings"]
    arch = mj["arch"]
    if cfg == "fd":
        counts = mj["class_counts"]
        data = [("Corpus", "FinGPT sentiment training set: 76,772 rows from Financial PhraseBank, FiQA, Twitter Financial News and News With GPT Instructions"),
                ("Labels", f"nine shades collapsed to three: negative {counts['negative']:,}, neutral {counts['neutral']:,}, positive {counts['positive']:,}"),
                ("Rows used", f"{mj['rows']:,} distinct texts after folding exact duplicates"),
                ("Split", f"{mj['rows_development']:,} development rows; {mj['rows_confirmation']:,} confirmation rows held out by text group before anything was fitted, scored once at the end")]
        dropout = f"dropout {c['dropout']}, input dropout {c['input_dropout']}, label smoothing {c['smoothing']}"
        conf = mj["confirmation"]
        scores = [("Cross-validated, 15 runs (5 grouped folds x 3 seeds)", f"accuracy {pct(mj['cv']['mean']['accuracy'])}, macro-F1 {f3(mj['cv']['mean']['macro_f1'])} (sd {mj['cv']['sd']['macro_f1']:.3f})"),
                  (f"Confirmation split, {mj['rows_confirmation']:,} rows", f"accuracy {pct(conf['metrics']['accuracy'])}, macro-F1 {f3(conf['metrics']['macro_f1'])} {ci(conf['bootstrap']['macro_f1_ci'])}, ECE {conf['metrics']['ece_15']:.3f}"),
                  (f"SEntFiN zero-shot, {n_sf:,} headlines", f"accuracy {pct(acc)}, macro-F1 {f3(f1)}")]
        p = lin["sentiment"]["paired"]["network_minus_linear_macro_f1"]
        scores.append(("Against FinGPT Linear, same 15 runs", f"network ahead by {p['mean']:+.3f} macro-F1 [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], p {p['p']:.1g}"))
        temps = f"{mj['temperature']:.3f} (ECE {mj['calibration']['ece_before']:.3f} to {mj['calibration']['ece_after']:.3f} out of fold)"
    else:
        data = [("Corpus", "EmpatheticDialogues: two-turn windows of conversations, in conversation order"),
                ("Labels", "emotion: the speaker's, 32 classes; sentiment: the FinGPT Network's reading of each window (negative 3,512, neutral 55,227, positive 5,827)"),
                ("Rows used", f"{mj['rows']:,} windows after folding duplicates"),
                ("Split", f"{mj['rows_development']:,} development rows; {mj['rows_confirmation']:,} confirmation rows held out by conversation before anything was fitted")]
        dropout = f"no dropout, task weight {c.get('task_weight', 1.0)} (loss = sentiment cross-entropy + emotion cross-entropy)"
        ct = mj["confirmation"]["tasks"]
        cv = mj["cv"]["mean"]
        scores = [("Cross-validated, 15 runs", f"sentiment accuracy {pct(cv['sentiment_accuracy'])}, macro-F1 {f3(cv['sentiment_macro_f1'])}; emotion accuracy {pct(cv['emotion_accuracy'])}, macro-F1 {f3(cv['emotion_macro_f1'])}"),
                  (f"Confirmation split, {mj['rows_confirmation']:,} rows", f"sentiment {pct(ct['sentiment']['metrics']['accuracy'])}, macro-F1 {f3(ct['sentiment']['metrics']['macro_f1'])} {ci(ct['sentiment']['bootstrap']['macro_f1_ci'])}; "
                                                                           f"emotion {pct(ct['emotion']['metrics']['accuracy'])}, macro-F1 {f3(ct['emotion']['metrics']['macro_f1'])} {ci(ct['emotion']['bootstrap']['macro_f1_ci'])}"),
                  (f"SEntFiN zero-shot, {n_sf:,} headlines", f"sentiment accuracy {pct(acc)}, macro-F1 {f3(f1)} (no finance label was ever seen)")]
        stage1 = load(results / "ed/rl/suite_standard_train/grpo_full/rl_summary.json")["before"]
        scores.append(("FinGPT confirmation rows, 4,317", f"accuracy {pct(stage1['held_accuracy'])}, macro-F1 {f3(stage1['held_macro_f1'])}"))
        for r in ("sentiment", "emotion"):
            p = lin[r]["paired"]["network_minus_linear_macro_f1"]
            scores.append((f"Against Dialogue Linear, {r}, same 15 runs", f"network minus linear {p['mean']:+.3f} macro-F1 [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], p {p['p']:.2g}"))
        temps = ", ".join(f"{t['name']} {t['temperature']:.3f}" for t in mj["tasks"])
    job = job_report(logs, f"{cfg}_nn_")
    start, end = sgd["final"][0][1], sgd["final_running_mean"][-1][1]
    details = [
        ("What it is", [("Model", f"a feed-forward network on the encoder's values: {arch}"), ("Weights", f"{info['params']:,}"), ("Encoder", ENCODER)]),
        ("Data", data),
        ("How it was chosen", [("Search", "34 configurations (32 sampled widths, depths and activations, the multinomial logit and the lecture's 64 ReLU then 32 tanh shape), successive halving on 2 folds with 8, 16 and 32 epochs"),
                               ("Selection", f"10 survivors on 15 runs each; the winner, {c['name']}, has the best mean macro-F1; every other survivor compared with it by the corrected resampled t-test"),
                               ("Final fit", f"{mj['epochs_final']} epochs on all {mj['rows_development']:,} development rows (the median best epoch of the 15 runs)")]),
        ("Stochastic gradient descent", [("Optimiser", f"AdamW, mini-batches of {c['batch']}, learning rate {c['lr']:g} (from a learning-rate range test), weight decay {c['weight_decay']}"),
                                         ("Schedule", f"{o['warmup_epochs']} warm-up epoch, then cosine decay to {o['cosine_floor']} of the rate; exponential moving average of the weights {o['ema']}; early stopping on validation cross-entropy, patience {o['patience']}"),
                                         ("Regularisation", dropout),
                                         ("Final fit", f"{sgd['total_steps']:,} gradient steps ({sgd['steps_per_epoch']} per epoch); batch cross-entropy {start:.3f} at the first step, {end:.3f} at the end (running mean of 50)")]),
        ("Calibration", [("Temperature", temps)]),
        ("Scores", scores),
        ("Cost", [("Search and final fit", f"{sel['minutes_total']} min on 16 CPU cores" + (f"; the whole job, with the predictions and the inspection, {job['minutes']} min, peak memory {job['memory']}" if job else "")),
                  ("Encoding", "each training text once, on one GPU in fp16 (FinGPT 1.8 min, dialogue 3.9 min); cached and reused by every model")]),
    ]
    curve = {"title": "Final fit: cross-entropy of each mini-batch (running mean of 50)", "x": [s for s, _ in sgd["final_running_mean"]][::4],
             "y": [v for _, v in sgd["final_running_mean"]][::4], "xlabel": "gradient step", "ylabel": "cross-entropy"}
    return details, curve


def stage_rows(results, path, name_of_start):
    s = load(results / path / "rl_summary.json")
    b, a = s["before"], s["after"]
    method = {"exact": "closed form of the leashed objective", "grpo": "GRPO"}[s["method"]]
    corpus = {"sentfin": "SEntFiN headlines", "train": "FinGPT development texts"}[s["adapt_set"]]
    which = {"all": "every headline of the training half", "wrong": "only the headlines it read wrongly in the training half"}.get(s.get("rows"), "")
    how = ("target on each row = the starting distribution tilted towards the label, softmax(log p_start + onehot(label) / beta), fitted by cross-entropy"
           if s["method"] == "exact" else "four sampled answers per text, reward +1 when right, advantage relative to the group, KL leash to the starting network")
    rows = [("Starts from", name_of_start), ("Method", f"{method}: {how}"),
            ("Trained on", f"{s['adapt_rows']:,} {corpus}" + (f", {which}" if which else "")),
            ("Held out", f"{s['held_out_rows']:,} rows never trained on, scored before and after"),
            ("Leash and retention", f"beta {s['beta']}; retention weight {s['beta_retain']} (a KL term on the network's own training rows that holds what it already read)"),
            ("Updates", f"AdamW, learning rate {s['lr']:g}, batch {s['batch']}, {s['epochs']} epochs, {s['steps']:,} gradient steps, gradient clip 1; {100 * s['parameters_changed_share']:.1f}% of the weights moved"),
            ("Time", f"{s['minutes']} min")]
    held = "SEntFiN held-out half" if s["adapt_set"] == "sentfin" else "FinGPT confirmation rows"
    scores = [(f"{held} ({b['held_rows']:,})", f"accuracy {pct(b['held_accuracy'])} to {pct(a['held_accuracy'])}, macro-F1 {f3(b['held_macro_f1'])} to {f3(a['held_macro_f1'])}"),
              ("Distance moved", f"KL to the starting network {a['held_kl_to_ref']:.3f} on the held-out rows; mean entropy {b['held_entropy']:.3f} to {a['held_entropy']:.3f}")]
    if "transfer_accuracy" in a:
        t = "SEntFiN, all headlines" if s["adapt_set"] == "train" else "FinGPT confirmation rows"
        scores.append((f"{t} ({b['transfer_rows']:,})", f"accuracy {pct(b['transfer_accuracy'])} to {pct(a['transfer_accuracy'])}, macro-F1 {f3(b['transfer_macro_f1'])} to {f3(a['transfer_macro_f1'])}"))
    if "home_sentiment_accuracy" in a and "home_emotion_macro_f1" not in a:
        scores.append((f"FinGPT confirmation rows ({b['home_rows']:,})", f"accuracy {pct(b['home_sentiment_accuracy'])} to {pct(a['home_sentiment_accuracy'])}, macro-F1 {f3(b['home_sentiment_macro_f1'])} to {f3(a['home_sentiment_macro_f1'])}"))
    for pre in ("dialogue", "home"):
        if f"{pre}_emotion_macro_f1" in a:
            scores.append((f"Dialogue confirmation rows ({b[pre + '_rows']:,})", f"sentiment agreement {pct(b[pre + '_sentiment_accuracy'])} to {pct(a[pre + '_sentiment_accuracy'])}; "
                                                                               f"emotion macro-F1 {f3(b[pre + '_emotion_macro_f1'])} to {f3(a[pre + '_emotion_macro_f1'])}"))
    return s, rows, scores


def rl_details(results, spec, names):
    base_name = "FinGPT Network" if spec["base"].startswith("fd") else "Dialogue Network"
    details, start = [], base_name
    curve = None
    for i, path in enumerate(spec["stages"], 1):
        s, rows, scores = stage_rows(results, path, start)
        title = f"Reinforcement, stage {i}" if len(spec["stages"]) > 1 else "Reinforcement"
        details.append((title, rows))
        details.append((f"Scores after stage {i}" if len(spec["stages"]) > 1 else "Scores", scores))
        start = names.get(path, start)
        log = list(csv.DictReader(open(results / path / "rl_log.csv", encoding="utf-8")))
        heldname = "SEntFiN held-out half" if s["adapt_set"] == "sentfin" else "FinGPT confirmation rows"
        curve = {"title": f"Held-out accuracy after each epoch ({heldname})", "x": [int(float(r["epoch"])) for r in log],
                 "y": [float(r["held_accuracy"]) for r in log], "xlabel": "epoch", "ylabel": "accuracy"}
    details.insert(0, ("What it is", [("Model", f"the {base_name} with the same layers and encoder, its weights moved by reinforcement learning against labels"),
                                      ("Reward", "+1 when the announced sentiment equals the label, 0 otherwise; the sentiment output's softmax is the policy"),
                                      ("Encoder", ENCODER)]))
    return details, curve


# ----- the linear heads

def save_linear(out, key, mj):
    arrays = {"mean": np.array(mj["mean"], np.float32), "sd": np.array(mj["sd"], np.float32)}
    readings = []
    for t in mj["tasks"]:
        arrays[f"out_{t['name']}_W"] = np.array(t["out"]["W"], np.float32)
        arrays[f"out_{t['name']}_b"] = np.array(t["out"]["b"], np.float32)
        readings.append({"name": t["name"], "classes": t["classes"], "temperature": float(t["temperature"])})
    np.savez_compressed(out / f"{key}.npz", **arrays)
    params = sum(arrays[f"out_{r['name']}_W"].size + arrays[f"out_{r['name']}_b"].size for r in readings)
    return {"acts": [], "widths": [], "readings": readings, "params": int(params)}


def linear_details(results, logs, cfg, info):
    S = load(results / f"{cfg}/train/linear/linear_summary.json")
    sets = load(results / f"{cfg}/sentfin/linear/metrics.json")["readings"]
    rows, scores = [], []
    for name, r in S["readings"].items():
        c = r["config"]
        label = name.capitalize()
        rows.append((label, f"C {c['C']} (L2 penalty strength 1/C), chosen on 2 folds of seed 0; final fit {c['iterations_final']} L-BFGS iterations; temperature {r['calibration']['temperature']:.3f}"))
        conf = r["confirmation"]
        scores.append((f"{label}, cross-validated (15 runs)", f"accuracy {pct(r['cv']['mean']['accuracy'])}, macro-F1 {f3(r['cv']['mean']['macro_f1'])} (sd {r['cv']['sd']['macro_f1']:.3f})"))
        scores.append((f"{label}, confirmation split ({conf['rows']:,})", f"accuracy {pct(conf['accuracy'])}, macro-F1 {f3(conf['macro_f1'])} {ci(conf['macro_f1_ci'])}"))
        p = r["paired"]["network_minus_linear_macro_f1"]
        scores.append((f"{label}, against the network on the same 15 runs", f"network minus linear {p['mean']:+.3f} macro-F1 [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], p {p['p']:.2g}"))
        if name in sets and "subsets" in sets[name]:
            a = sets[name]["subsets"]["all"]
            scores.append((f"{label}, SEntFiN zero-shot ({a['rows']:,})", f"accuracy {pct(a['accuracy'])}, macro-F1 {f3(a['macro_f1'])}; the network on the same headlines {pct(a['network']['accuracy'])}"))
    corpus = "FinGPT sentiment training set, the same rows and folds as the FinGPT Network" if cfg == "fd" else "EmpatheticDialogues windows, the same rows and folds as the Dialogue Network"
    first = next(iter(S["readings"].values()))
    details = [
        ("What it is", [("Model", "multinomial logistic regression on the encoder's standardised values: one linear layer per reading, no hidden layer"),
                        ("Weights", f"{info['params']:,}"), ("Encoder", ENCODER)]),
        ("Data", [("Corpus", corpus), ("Split", f"{first['rows_development']:,} development rows, {first['rows_confirmation']:,} confirmation rows (the networks' own split)")]),
        ("Training", [("Method", "maximum likelihood with an L2 (ridge) penalty, solved by L-BFGS on the full development set: a quasi-Newton method, no stochastic gradient descent and no mini-batches"),
                      ("Penalty", "C from 0.001, 0.003, 0.01, 0.03, 0.1 on the first 2 folds of seed 0, then the chosen C on all 15 folds x seeds")] + rows),
        ("Scores", scores),
        ("Cost", [("Run", f"{S['minutes_total']} min for {S['fits']} fits on {S['threads']} CPU cores, peak memory {S['peak_rss_mb'] / 1024:.1f} GB")]),
    ]
    return details, None


# ----- the word models

def save_words(out, key, mj):
    vocab = "\n".join(mj["vocab"]).encode("utf-8")
    np.savez_compressed(out / f"{key}.npz", vocab=np.frombuffer(vocab, np.uint8), texts=np.array(mj["texts"], np.int32),
                        coef=np.array(mj["coef"], np.float32), intercept=np.array(mj["intercept"], np.float32))
    return {"acts": [], "widths": [], "readings": [{"name": "label", "classes": mj["class_names"], "temperature": 1.0}],
            "params": int(mj["parameters"]), "vocabulary": int(mj["vocabulary"]), "trigger": float(mj["trigger_threshold"])}


def words_details(results, src, info):
    S = load(results / src / "bow_summary.json")
    conf = S["confirmation"]
    corpus = {"fingpt": "FinGPT sentiment training set, the nine shades kept (the FinGPT Network's rows and folds)",
              "sentfin": "SEntFiN 1.0 headlines, three sentiments, a grouped 15% split made for this model",
              "empathetic": "EmpatheticDialogues windows, the 32 emotions (the Dialogue Network's rows and folds)"}[src.split("/")[1]]
    scores = [("Cross-validated (15 runs)", f"accuracy {pct(S['cv_accuracy'])}, macro-F1 {f3(S['cv_macro_f1'])} (sd {S['cv_macro_f1_sd']:.3f})"),
              (f"Confirmation split ({conf['rows']:,})", f"accuracy {pct(conf['accuracy'])}, macro-F1 {f3(conf['macro_f1'])} {ci(conf['macro_f1_ci'])}")]
    if conf.get("polarity"):
        scores.append(("Same rows, shades summed to three", f"accuracy {pct(conf['polarity']['accuracy'])}, macro-F1 {f3(conf['polarity']['macro_f1'])}"))
    for s, r in S.get("extras", {}).items():
        scores.append((f"SEntFiN zero-shot ({r['labelled_rows']:,}), shades summed", f"accuracy {pct(r['accuracy'])}, macro-F1 {f3(r['macro_f1'])}; "
                                                                                  f"{100 * r['unknown_words']['share_of_word_occurrences_unknown']:.0f}% of its word occurrences were never seen in training"))
    u = conf["unknown_words"]
    t = S["timings_seconds"]
    details = [
        ("What it is", [("Model", "multinomial logit on word indicators: each distinct word of the text counts once; P(class) = softmax(intercept + sum of the known words' coefficients)"),
                        ("Identification", "by the ridge penalty: every class keeps its own coefficients and no base category is fixed at zero"),
                        ("Parameters", f"{info['params']:,} ({len(info['readings'][0]['classes'])} classes x {info['vocabulary']:,} words, plus intercepts)"),
                        ("Encoder", "none: the model reads the words themselves (lower case, letters a to z, an inner apostrophe allowed)")]),
        ("Data", [("Corpus", corpus), ("Rows", f"{S['rows']:,}; {S['rows_development']:,} development, {S['rows_confirmation']:,} confirmation"),
                  ("Vocabulary", f"{S['vocabulary']:,} words from the development rows; {S['words_in_one_text']:,} of them appear in one text only"),
                  ("Unknown words", f"a word outside the vocabulary has no coefficient: it is N/A and adds nothing. On the confirmation rows {u['rows_with_an_unknown_word']:,} texts had at least one")]),
        ("Training", [("Method", "maximum likelihood with an L2 penalty, solved by L-BFGS on the full sparse matrix: no stochastic gradient descent"),
                      ("Penalty", f"C chosen by grouped cross-validation from 0.03, 0.1, 0.3, 1, 3: C {S['C']:g}; final fit {S['iterations_final']} iterations"),
                      ("Trigger words", f"{S['trigger_words']:,} words whose answer on their own puts more than {int(100 * S['trigger_threshold'])}% on one class")]),
        ("Scores", scores),
        ("Cost", [("Run", f"{S['minutes']} min ({S['fits']} fits; cross-validation {t['cross_validation']:.0f} s, final fit {t['final_fit']:.1f} s), peak memory {S['peak_rss_mb']:,} MB")]),
    ]
    return details, None


def caption(key):
    c = {"fingpt_network": "FinGPT 79.8%, SEntFiN 73.5%", "fingpt_network_sentfin_all": "SEntFiN 81.9%, FinGPT 79.7%",
         "fingpt_network_sentfin_mistakes": "SEntFiN 80.9%, FinGPT 79.7%", "fingpt_network_sentfin_grpo": "SEntFiN 83.5%, FinGPT 77.1%",
         "dialogue_network": "sentiment + 32 emotions; SEntFiN 74.5%", "dialogue_network_fingpt": "FinGPT 78.6%, SEntFiN 74.6%",
         "dialogue_network_fingpt_sentfin_mistakes": "SEntFiN 75.6%, FinGPT 78.6%", "dialogue_network_fingpt_sentfin_all": "SEntFiN 76.9%, FinGPT 78.1%",
         "dialogue_network_fingpt_sentfin_grpo": "SEntFiN 83.0%, FinGPT 74.3%", "fingpt_linear": "FinGPT 78.2%, SEntFiN 71.4%",
         "dialogue_linear": "sentiment + 32 emotions; SEntFiN 74.3%", "words_fingpt": "nine shades; FinGPT 71.4% summed to three",
         "words_sentfin": "three sentiments; 79.9% on SEntFiN itself", "words_dialogue": "32 emotions; macro-F1 0.308"}
    return c[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage_names = {"ed/rl/suite_standard_train/grpo_full": "Dialogue Network + FinGPT"}
    catalog = []
    for spec in MODELS:
        key, kind = spec["key"], spec["kind"]
        if kind == "network":
            src = spec["stages"][-1] if spec.get("stages") else spec["base"]
            info = save_network(out, key, load(results / src / "nn_model.json"))
            details, curve = (rl_details(results, spec, stage_names) if spec.get("stages") else base_details(results, args.logs, spec["base"][:2], info))
        elif kind == "linear":
            info = save_linear(out, key, load(results / f"{spec['src']}/train/linear/linear_model.json"))
            details, curve = linear_details(results, args.logs, spec["src"], info)
        else:
            mj = load(results / spec["src"] / "bow_model.json")
            info = save_words(out, key, mj)
            details, curve = words_details(results, spec["src"], info)
        entry = {"key": key, "name": spec["name"], "tag": spec["tag"], "kind": kind, "encoder": kind != "words", "caption": caption(key), **info,
                 "details": [{"title": t, "rows": [[a, b] for a, b in rows]} for t, rows in details], "curve": curve}
        if key == "words_fingpt":
            entry["polarity"] = SHADES_TO_POLARITY
        catalog.append(entry)
        size = (out / f"{key}.npz").stat().st_size
        print(f"{spec['name']} [{spec['tag']}]: {info['params']:,} parameters, {size / 2**20:.1f} MB")
    (out / "catalog.json").write_text(json.dumps(catalog, indent=1), encoding="utf-8")
    print(f"{len(catalog)} models -> {out}")


if __name__ == "__main__":
    main()
