"""Sentence readings from fourteen sentiment and emotion models (Streamlit).

    pip install -r requirements.txt
    streamlit run app.py

Pick a model on the left and type a sentence. The networks and the linear heads read the 2,048
values of the frozen bge-large-en-v1.5 encoder (downloaded on first use, about 1.3 GB); the word
logits read the words themselves and never load the encoder. models/ holds the weights and the
facts each model shows; build_models.py writes it from the pipeline's result files.
"""
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sentiment prediction", page_icon="💬")
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
ENCODER = os.environ.get("SENTIMENT_ENCODER", "BAAI/bge-large-en-v1.5")  # a local folder with the weights also works
MAX_LEN, OVERLAP = 512, 32

SENTIMENT = ["negative", "neutral", "positive"]
SENT_COL = {"negative": "#e5484d", "neutral": "#8b5cf6", "positive": "#1f9d55"}
ICON = {"negative": "🔴", "neutral": "🟣", "positive": "🟢"}
SHADE_COL = {"strong negative": "#b4232a", "negative": "#e5484d", "moderately negative": "#ef7f82", "mildly negative": "#f6b3b5",
             "neutral": "#8b5cf6", "mildly positive": "#a6dcb9", "moderately positive": "#5fbf83", "positive": "#1f9d55", "strong positive": "#11703a"}
ACCENT, NEG = "#7c4df5", "#e5484d"


# ---------- text, exactly as the training texts were cleaned ----------
_TAIL = chr(0x80) + "-" + chr(0xbf) + "".join(map(chr, (0x20ac, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021, 0x02c6, 0x2030, 0x0160, 0x2039, 0x0152,
                                                        0x017d, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0x02dc, 0x2122, 0x0161, 0x203a,
                                                        0x0153, 0x017e, 0x0178)))
_MOJIBAKE = re.compile("(?:" + chr(0xc2) + "[ " + _TAIL + "]|[" + chr(0xc3) + chr(0xe2) + "][" + _TAIL + "])")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def fix_mojibake(s):
    """UTF-8 text that was decoded as Windows-1252 and saved again; left alone unless the tell-tale pairs are there."""
    if not _MOJIBAKE.search(s):
        return s
    s = s.replace(chr(0xc2) + " ", " ")
    if not _MOJIBAKE.search(s):
        return s
    for errors in ("strict", "ignore"):
        try:
            fixed = s.encode("cp1252", errors=errors).decode("utf-8", errors=errors)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if len(fixed) >= 0.7 * len(s):
            return fixed
    return s


def clean(text):
    s = unicodedata.normalize("NFC", fix_mojibake(text))
    s = s.replace(chr(0x200b), "").replace(chr(0xfeff), "").replace(chr(0xa0), " ").replace("_comma_", ",")
    return re.sub(r"\s+", " ", s).strip()


def words(text):
    """Distinct words in order of first appearance, the word logits' tokeniser."""
    seen, out = set(), []
    for w in _WORD.findall(text.lower().replace(chr(0x2019), "'")):
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# ---------- models ----------
@st.cache_resource(show_spinner=False)
def catalog():
    return {m["key"]: m for m in json.loads((MODELS / "catalog.json").read_text(encoding="utf-8"))}


@st.cache_resource(show_spinner=False)
def load_model(key):
    m = catalog()[key]
    with np.load(MODELS / f"{key}.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if m["kind"] == "words":
        vocab = a["vocab"].tobytes().decode("utf-8").split("\n")
        return {"vocab": vocab, "index": {w: i for i, w in enumerate(vocab)}, "texts": a["texts"],
                "coef": a["coef"].astype(np.float64), "intercept": a["intercept"].astype(np.float64)}
    def f64(k):
        return a[k].astype(np.float64)
    return {"mean": f64("mean"), "sd": f64("sd"),
            "layers": [(f64(f"W{i}"), f64(f"b{i}"), act) for i, act in enumerate(m["acts"])],
            "outs": {r["name"]: (f64(f"out_{r['name']}_W"), f64(f"out_{r['name']}_b")) for r in m["readings"]}}


@st.cache_resource(show_spinner=False)
def encoder():
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(ENCODER), AutoModel.from_pretrained(ENCODER, add_pooling_layer=False).eval()


@st.cache_data(show_spinner=False, max_entries=500)
def encode(text):
    """[CLS] vector then the mean over the tokens of bge-large's last layer, 2,048 values. A text
    longer than one 512-token window is read in overlapping windows averaged by length, as in training."""
    import torch
    tok, enc = encoder()
    ids = tok(text, add_special_tokens=False, verbose=False)["input_ids"]
    body, step = MAX_LEN - 2, MAX_LEN - 2 - OVERLAP
    total, weight, windows = np.zeros(2 * enc.config.hidden_size), 0, 0
    with torch.no_grad():
        for start in range(0, max(len(ids) - OVERLAP, 1), step):
            window = [tok.cls_token_id] + ids[start:start + body] + [tok.sep_token_id]
            h = enc(input_ids=torch.tensor([window])).last_hidden_state[0]
            total += torch.cat([h[0], h.mean(0)]).numpy().astype(np.float64) * len(window)
            weight += len(window)
            windows += 1
    return total / weight, len(ids), windows


ACT = {
    "relu": lambda x: np.maximum(x, 0),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "silu": lambda x: x / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0))),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}


def softmax(z):
    p = np.exp(z - z.max())
    return p / p.sum()


def run_head(model, meta, x):
    z = (x - model["mean"]) / model["sd"]
    h, layers = z, []
    with np.errstate(over="ignore"):
        for W, b, act in model["layers"]:
            pre = W @ h + b
            h = ACT[act](pre)
            layers.append((pre, h))
    out = {}
    for r in meta["readings"]:
        W, b = model["outs"][r["name"]]
        logits = W @ h + b
        out[r["name"]] = {"classes": r["classes"], "T": r["temperature"], "logits": logits, "p": softmax(logits / r["temperature"])}
    return {"x": x, "z": z, "layers": layers, "h": h, "out": out}


def run_words(model, meta, text):
    ws = words(text)
    rows, z = [], model["intercept"].copy()
    for w in ws:
        j = model["index"].get(w)
        if j is None:
            rows.append({"word": w, "j": None})
        else:
            z += model["coef"][:, j]
            rows.append({"word": w, "j": j, "texts": int(model["texts"][j]), "p": softmax(model["intercept"] + model["coef"][:, j])})
    classes = meta["readings"][0]["classes"]
    return {"words": rows, "known": sum(r["j"] is not None for r in rows), "logits": z, "p": softmax(z), "classes": classes}


# ---------- display ----------
def pct(p, d=1):
    """A percentage that does not round to 0% or 100% unless the value is exactly that."""
    s = f"{p:.{d}%}"
    if p > 0 and s == f"{0:.{d}%}":
        return f"<{10 ** -(d + 2):.{d}%}"
    if p < 1 and s == f"{1:.{d}%}":
        return f">{1 - 10 ** -(d + 2):.{d}%}"
    return s


def table(data, rows=25):
    """A dataframe tall enough for all its rows, up to `rows`; longer ones scroll."""
    n = len(next(iter(data.values())))
    st.dataframe(data, hide_index=True, width="stretch", height=35 * (min(n, rows) + 1) + 3)


def bars(labels, values, colours, height=None):
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colours, text=[pct(v) for v in values], textposition="outside"))
    fig.update_layout(height=height or 60 + 34 * len(labels), margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(range=[0, 1.12], tickformat=".0%"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")


def top_label(classes, p):
    k = int(np.argmax(p))
    return classes[k], float(p[k])


def show_sentiment(classes, p, title="Sentiment"):
    by = dict(zip(classes, p))
    label, conf = top_label(classes, p)
    st.metric(title, f"{ICON.get(label, '')} {label}", f"{pct(conf, 0)} confident", delta_color="off")
    bars(SENTIMENT, [float(by[c]) for c in SENTIMENT], [SENT_COL[c] for c in SENTIMENT], height=200)


def show_emotion(classes, p, title="Emotion"):
    order = np.argsort(-p)
    label, conf = classes[order[0]], float(p[order[0]])
    st.metric(title, label, f"{pct(conf, 0)} confident, of {len(classes)} emotions", delta_color="off")
    top = order[:8]
    labels = [classes[k] for k in top] + [f"the other {len(classes) - len(top)}"]
    values = [float(p[k]) for k in top] + [float(p[order[len(top):]].sum())]
    bars(labels, values, [ACCENT] * len(top) + ["#c9c3dc"])


def show_shades(classes, p, polarity):
    label, conf = top_label(classes, p)
    summed = {c: 0.0 for c in SENTIMENT}
    for c, v in zip(classes, p):
        summed[polarity[c]] += float(v)
    pol = max(summed, key=summed.get)
    c1, c2 = st.columns(2)
    c1.metric("Shade", label, f"{pct(conf, 0)} confident, of 9 shades", delta_color="off")
    c2.metric("Polarity, the shades summed", f"{ICON[pol]} {pol}", pct(summed[pol], 0), delta_color="off")
    bars(classes, [float(v) for v in p], [SHADE_COL.get(c, ACCENT) for c in classes])


def show_reading(meta, res):
    if meta["kind"] == "words":
        classes, p = res["classes"], res["p"]
        if res["known"] == 0:
            st.metric("Reading", "N/A")
        elif meta.get("polarity"):
            show_shades(classes, p, meta["polarity"])
        elif classes == SENTIMENT:
            show_sentiment(classes, p)
        else:
            show_emotion(classes, p)
        show_word_table(meta, res)
        return
    for name, r in res["out"].items():
        (show_sentiment if name == "sentiment" else show_emotion)(r["classes"], r["p"])


def show_word_table(meta, res):
    n, k = len(res["words"]), res["known"]
    classes = res["classes"]
    if n == 0:
        st.warning("No words to read: this model sees only the letters a to z, so it gives no reading.")
        return
    if k == 0:
        st.warning(("The word was never seen in training, so it is N/A" if n == 1 else f"None of the {n} words was seen in training, so all are N/A")
                   + " and this model gives no reading: a word logit can read only the words it was trained on.")
    elif k < n:
        st.info(f"{n - k} of {n} words {'are' if n - k > 1 else 'is'} N/A (never seen in training) and add{'' if n - k > 1 else 's'} nothing; "
                f"the reading rests on the other {k}.")
    thr = meta.get("trigger", 0.5)
    st.caption("Each word on its own: what the model answers for a text made of that one word, softmax(intercept + the word's coefficients). "
               f"A trigger word puts more than {thr:.0%} on one class.")
    data = {"word": [r["word"] for r in res["words"]], "texts in training": [f"{r['texts']:,}" if r["j"] is not None else "N/A" for r in res["words"]],
            "trigger": ["N/A" if r["j"] is None else classes[int(np.argmax(r["p"]))] if r["p"].max() > thr else "no" for r in res["words"]]}
    if len(classes) <= 9:
        for i, c in enumerate(classes):
            data[c] = [pct(r["p"][i]) if r["j"] is not None else "N/A" for r in res["words"]]
    else:
        for rank in range(3):
            col_c, col_p = ["first", "second", "third"][rank], f"p{rank + 1}"
            data[col_c], data[col_p] = [], []
            for r in res["words"]:
                if r["j"] is None:
                    data[col_c].append("N/A")
                    data[col_p].append("N/A")
                else:
                    i = int(np.argsort(-r["p"])[rank])
                    data[col_c].append(classes[i])
                    data[col_p].append(pct(r["p"][i]))
    table(data)


# ---------- the calculation ----------
def feature_part(j, half):
    return "[CLS] vector" if j < half else "token mean"


def show_encoder_step(res, n_tokens, n_windows):
    x = res["x"]
    half = len(x) // 2
    st.markdown(f"**1. Encoder.** bge-large read {n_tokens} tokens in {n_windows} window{'s' if n_windows > 1 else ''} and gave {len(x):,} values: "
                f"the [CLS] vector (positions 0 to {half - 1}) and the mean over the tokens ({half} to {len(x) - 1}). "
                f"Range {x.min():.3f} to {x.max():.3f}, length {np.linalg.norm(x):.2f}.")
    if st.toggle("Show all 2,048 values", key="all_values"):
        table({"position": np.arange(len(x)), "part": [feature_part(j, half) for j in range(len(x))], "value": np.round(x, 5)}, rows=12)


def show_standardise_step(model, res):
    z, x = res["z"], res["x"]
    half = len(x) // 2
    top = np.argsort(-np.abs(z))[:10]
    st.markdown("**2. Standardise.** z = (x - mean) / sd, with the mean and sd of the training rows, so every value is in units of its own spread. "
                f"{int((np.abs(z) > 3).sum())} of {len(z):,} values are more than 3 sd from their training average; the ten furthest:")
    table({"position": top, "part": [feature_part(j, half) for j in top], "value x": np.round(x[top], 4),
           "training mean": np.round(model["mean"][top], 4), "training sd": np.round(model["sd"][top], 4), "z": np.round(z[top], 2)})


def show_output_step(meta, model, res, step):
    for name, r in res["out"].items():
        classes, logits, T, p = r["classes"], r["logits"], r["T"], r["p"]
        order = np.argsort(-p)[: min(len(classes), 8)]
        source = f"logits = W h + b over the {len(res['h'])} units of the last hidden layer" if meta["acts"] else \
            f"logits = W z + b straight from the {len(res['h']):,} standardised values, no hidden layer"
        st.markdown(f"**{step}. Output, {name}.** {source}, divided by the temperature T = {T:.3f}, then softmax"
                    + (f" (the top {len(order)} of {len(classes)} shown)" if len(order) < len(classes) else "") + ":")
        table({"class": [classes[k] for k in order], "logit": np.round(logits[order], 3), "logit / T": np.round(logits[order] / T, 3),
               "probability": [pct(p[k], 2) for k in order]})
        win = int(order[0])
        W, b = model["outs"][name]
        contrib = W[win] * res["h"]
        idx = np.argsort(-np.abs(contrib))[:8]
        what = "unit of the last hidden layer" if meta["acts"] else "standardised input value"
        st.caption(f"What pushed {classes[win]} most: each {what} times its weight to {classes[win]} (bias {b[win]:.3f}).")
        table({"unit" if meta["acts"] else "position": idx, "value": np.round(res["h"][idx], 4), "weight": np.round(W[win, idx], 4),
               "contribution": np.round(contrib[idx], 4)})


def show_network_calculation(meta, model, res, n_tokens, n_windows):
    show_encoder_step(res, n_tokens, n_windows)
    show_standardise_step(model, res)
    step = 3
    for i, ((pre, h), (W, _, act)) in enumerate(zip(res["layers"], model["layers"])):
        active = int((h > 0).sum())
        top = np.argsort(-np.abs(h))[:8]
        st.markdown(f"**{step}. Hidden layer {i + 1}.** {W.shape[0]} units, each a weighted sum of the {W.shape[1]:,} values before it plus a bias, a = W h + b, "
                    f"then {act.replace('_', ' ')}. {active} of {len(h)} units come out positive; the eight largest:")
        table({"unit": top, "weighted sum a": np.round(pre[top], 4), f"after {act.replace('_', ' ')}": np.round(h[top], 4)})
        step += 1
    show_output_step(meta, model, res, step)


def show_word_calculation(meta, model, res):
    classes, z, p = res["classes"], res["logits"], res["p"]
    cols = list(range(len(classes))) if len(classes) <= 9 else [int(k) for k in np.argsort(-p)[:5]]
    st.markdown("Each class gets a score: its intercept plus the coefficient of every known word (a word counts once however often it appears). "
                "Softmax turns the scores into probabilities. N/A words have no coefficient and add nothing."
                + ("" if len(cols) == len(classes) else f" The {len(cols)} highest-scoring of the {len(classes)} classes are shown."))
    rows = [("intercept", model["intercept"][cols])] + [(r["word"], model["coef"][cols, r["j"]]) for r in res["words"] if r["j"] is not None]
    data = {"term": [t for t, _ in rows] + ["score (sum)", "probability"]}
    for i, k in enumerate(cols):
        data[classes[k]] = [f"{v[i]:+.3f}" for _, v in rows] + [f"{z[k]:+.3f}", pct(p[k], 2)]
    table(data, rows=40)
    unknown = [r["word"] for r in res["words"] if r["j"] is None]
    if unknown:
        st.caption("N/A, left out of the sum: " + ", ".join(unknown))
    if not res["known"]:
        st.caption("With no known word the sum is the intercepts alone, the prior the model gives every such text, so the reading above is N/A.")


# ---------- the network drawn ----------
DIVERGING = [[0, NEG], [0.5, "#efeaf9"], [1, ACCENT]]


def pathway(model, res, out_nodes, n_in=12, n_unit=10):
    """The units that carried most of this answer, traced back from the drawn outputs: in each layer,
    the units whose weighted values fed the units kept after them most strongly."""
    hs = [h for _, h in res["layers"]]
    rel = sum(np.abs(model["outs"][name][0][k] * res["h"]) for name, k in out_nodes)
    keep = [None] * len(hs)
    for li in range(len(hs) - 1, -1, -1):
        keep[li] = np.argsort(-rel)[:n_unit]
        below = res["z"] if li == 0 else hs[li - 1]
        rel = np.abs(model["layers"][li][0][keep[li]] * below).sum(0)
    return np.argsort(-rel)[:n_in], keep


def spread(n, step=None, lo=0.2, hi=1.0):
    """n heights from the top down, evenly spaced and centred in [lo, hi]."""
    if n == 1:
        return np.array([(lo + hi) / 2])
    step = min((hi - lo) / (n - 1), step or 1.0)
    return (lo + hi) / 2 + step * ((n - 1) / 2 - np.arange(n))


def strongest(W, values, src, dst, n):
    """For each drawn unit d, its n largest contributions W[d, s] * value[s] from the drawn units s."""
    out = []
    for d in dst:
        c = W[d, src] * values[src]
        out += [(int(src[i]), int(d), float(c[i])) for i in np.argsort(-np.abs(c))[:n]]
    return out


def strip(fig, x, values, label, limit=None):
    """Every unit of a layer as one thin bar under its column, sorted from lowest to highest."""
    order = np.argsort(values, kind="stable")
    dx = 0.64 / len(values)
    m = limit or float(np.abs(values).max()) or 1.0
    fig.add_trace(go.Heatmap(z=[values[order]], customdata=[order], x0=x - 0.32 + dx / 2, dx=dx, y0=0.07, dy=0.07, zmin=-m, zmax=m,
                             colorscale=DIVERGING, showscale=False, hovertemplate=label + ", unit %{customdata}: %{z:.3f}<extra></extra>"))


def network_figure(meta, model, res):
    z, L = res["z"], len(res["layers"])
    half = len(z) // 2
    out_nodes = []
    for name, r in res["out"].items():
        top = np.argsort(-r["p"])
        out_nodes += [(name, int(k)) for k in (top if len(top) <= 9 else top[:5])]
    keep_in, keep = pathway(model, res, out_nodes)
    fig = go.Figure()
    edges = []  # (from x, from y, to x, to y, contribution, transition)

    # inputs, highest z at the top
    order_in = keep_in[np.argsort(-z[keep_in])]
    y_prev = dict(zip(order_in.tolist(), spread(len(order_in))))
    fig.add_trace(go.Scatter(x=[0] * len(order_in), y=list(y_prev.values()), mode="markers+text", cliponaxis=False, textposition="middle left",
                             text=[f"{'[CLS]' if j < half else 'mean'} {j}  z {z[j]:+.2f}" for j in order_in], textfont=dict(size=11, color="#6b6b76"),
                             marker=dict(size=9, color=[ACCENT if z[j] > 0 else NEG for j in order_in], line=dict(width=1, color="white")),
                             hovertext=[f"input {j} ({feature_part(j, half)})<br>value {res['x'][j]:.4f}, z {z[j]:+.3f}" for j in order_in],
                             hoverinfo="text", showlegend=False))
    strip(fig, 0, z, "input z", limit=3.0)

    # hidden layers, highest output at the top
    vals_prev, src = z, order_in
    for li, ((pre, h), (W, _, act)) in enumerate(zip(res["layers"], model["layers"])):
        order = keep[li][np.argsort(-h[keep[li]])]
        y_here = dict(zip(order.tolist(), spread(len(order))))
        edges += [(li, y_prev[s], li + 1, y_here[d], c, li) for s, d, c in strongest(W, vals_prev, src, order, 3)]
        m = float(np.abs(h[order]).max()) or 1.0
        fig.add_trace(go.Scatter(x=[li + 1] * len(order), y=list(y_here.values()), mode="markers",
                                 marker=dict(size=7 + 9 * np.abs(h[order]) / m, color=[ACCENT if h[u] > 0 else NEG for u in order], line=dict(width=1, color="white")),
                                 hovertext=[f"layer {li + 1}, unit {u}<br>weighted sum {pre[u]:.4f}<br>after {act.replace('_', ' ')} {h[u]:.4f}" for u in order],
                                 hoverinfo="text", showlegend=False))
        strip(fig, li + 1, h, f"layer {li + 1}")
        vals_prev, src, y_prev = h, order, y_here

    # outputs, each reading's classes from likeliest down, a gap between readings
    slots, names = [], list(res["out"])
    for name in names:
        if slots:
            slots.append(None)
        slots += [(name, k) for n, k in out_nodes if n == name]
    y_out = {s: y for s, y in zip(slots, spread(len(slots), step=0.2)) if s is not None}
    for name, k in out_nodes:
        W, _ = model["outs"][name]
        edges += [(L, y_prev[s], L + 1, y_out[(name, k)], c, L) for s, _, c in strongest(W, vals_prev, src, [k], 3)]
    info = [(res["out"][n]["classes"][k], float(res["out"][n]["p"][k]), float(res["out"][n]["logits"][k]), n) for n, k in out_nodes]
    fig.add_trace(go.Scatter(x=[L + 1] * len(out_nodes), y=[y_out[s] for s in out_nodes], mode="markers+text", cliponaxis=False, textposition="middle right",
                             text=[f"{c} {pct(p)}" for c, p, _, _ in info], textfont=dict(size=11, color="#3d3d45"),
                             marker=dict(size=[9 + 22 * p for _, p, _, _ in info], color=[SENT_COL.get(c, ACCENT) if n == "sentiment" else ACCENT for c, _, _, n in info],
                                         line=dict(width=1, color="white")),
                             hovertext=[f"{n}: {c}<br>logit {lg:.3f}, probability {pct(p, 2)}" for c, p, lg, n in info], hoverinfo="text", showlegend=False))

    # lines: purple raises the unit it feeds, red lowers it; width by size within each step between layers
    big = {t: max(abs(e[4]) for e in edges if e[5] == t) or 1.0 for t in {e[5] for e in edges}}
    lines = []
    for sign, colour in ((1, ACCENT), (-1, NEG)):
        for lo, hi, width in ((0, 0.25, 0.8), (0.25, 0.5, 1.6), (0.5, 0.75, 2.6), (0.75, 1.01, 4.0)):
            xs, ys = [], []
            for x0, y0, x1, y1, c, t in edges:
                if np.sign(c) == sign and lo <= abs(c) / big[t] < hi:
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
            if xs:
                lines.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=width, color=colour), opacity=0.5, hoverinfo="skip", showlegend=False))
    if lines:
        fig.add_traces(lines)
        fig.data = fig.data[-len(lines):] + fig.data[:-len(lines)]

    ticks = ["input<br>2,048 values"] + [f"layer {i + 1}<br>{len(h)} {act.replace('_', ' ')}" for i, ((_, h), (_, _, act)) in enumerate(zip(res["layers"], model["layers"]))]
    ticks.append("output<br>" + " + ".join(names))
    fig.update_layout(height=520, margin=dict(l=140, r=130, t=10, b=50), plot_bgcolor="rgba(0,0,0,0)", hovermode="closest",
                      xaxis=dict(tickvals=list(range(L + 2)), ticktext=ticks, tickfont=dict(size=11), showgrid=False, zeroline=False,
                                 range=[-0.4, L + 1.4], fixedrange=True),
                      yaxis=dict(visible=False, range=[0, 1.06], fixedrange=True))
    return fig


# ---------- page ----------
STYLE = """<style>
.stApp { background: linear-gradient(160deg, #efe8ff 0%, #f8f6ff 35%, #fdfcff 60%, #e8f1ff 100%); background-attachment: fixed; }
.stApp header[data-testid="stHeader"] { background: transparent; }
.stApp h1 { background: linear-gradient(90deg, #6d3cf0 0%, #b5179e 55%, #3a7bfd 100%); -webkit-background-clip: text; background-clip: text;
            color: transparent; font-weight: 800; letter-spacing: -0.02em; padding-bottom: 0.1rem; }
.hero-sub { color: #5d5873; margin: -0.6rem 0 1.4rem; font-size: 1.02rem; }
.st-key-card_input, .st-key-card_result, .st-key-card_more {
    background: rgba(255, 255, 255, 0.82); border: 1px solid rgba(124, 77, 245, 0.14) !important; border-radius: 22px !important;
    box-shadow: 0 10px 30px -12px rgba(76, 40, 170, 0.22); padding: 1.3rem 1.4rem 1.1rem; backdrop-filter: blur(6px); }
.st-key-card_result { border-top: 4px solid transparent !important;
    background: linear-gradient(#ffffff, #ffffff) padding-box, linear-gradient(90deg, #7c4df5, #c026d3, #3a7bfd) border-box; }
.st-key-predict button { background: linear-gradient(90deg, #7c4df5 0%, #b5179e 100%); border: none; border-radius: 999px;
    padding: 0.5rem 2.2rem; font-weight: 600; box-shadow: 0 6px 18px -6px rgba(124, 77, 245, 0.6); transition: transform .15s, box-shadow .15s; }
.st-key-predict button:hover { transform: translateY(-1px); box-shadow: 0 10px 22px -6px rgba(124, 77, 245, 0.7); color: white; }
.stTextArea textarea, .stSelectbox [data-baseweb="select"] > div { border-radius: 14px !important; }
[data-testid="stMetric"] { background: linear-gradient(135deg, #f4efff 0%, #eef4ff 100%); border-radius: 16px; padding: 0.8rem 1.1rem; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>"""

cat = catalog()
st.html(STYLE)
st.title("Sentiment prediction")
st.html('<p class="hero-sub">Pick one of fourteen models, type a sentence and see how it reads it.</p>')

with st.container(border=True, key="card_input"):
    key = st.selectbox("Model", list(cat), format_func=lambda k: f"{cat[k]['name']} [{cat[k]['tag']}]")
    meta = cat[key]
    st.caption("Encoder: bge-large-en-v1.5, frozen, 2,048 values per text" if meta["encoder"] else "Encoder: none. This model reads the words themselves.")
    st.caption(f"{meta['caption']}. Percentages are accuracy on rows the model never trained on.")
    sentence = st.text_area("Sentence", placeholder="e.g. Shares jumped after the company beat earnings forecasts.", height=100)
    if st.button("Predict", type="primary", key="predict"):
        if not sentence.strip():
            st.warning("Type a sentence first.")
        else:
            st.session_state["sentence"] = sentence.strip()

text = st.session_state.get("sentence")
res, n_tokens, n_windows = None, 0, 0
if text:
    model = load_model(key)
    if meta["encoder"]:
        try:
            with st.spinner("Encoding with bge-large (the first run downloads it, about 1.3 GB) ..."):
                x, n_tokens, n_windows = encode(clean(text))
        except OSError as e:
            st.error(f"The encoder could not be loaded: {e}")
        else:
            res = run_head(model, meta, x)
    else:
        res = run_words(model, meta, clean(text))

if res is not None:
    with st.container(border=True, key="card_result"):
        st.subheader("Result")
        if text != sentence.strip():
            st.caption(f'Result for: "{text}"')
        show_reading(meta, res)
        if st.toggle("Show the calculation", key="calculation"):
            if meta["kind"] == "words":
                show_word_calculation(meta, model, res)
            else:
                show_network_calculation(meta, model, res, n_tokens, n_windows)

with st.container(border=True, key="card_more"):
    if st.toggle("Model details", key="details"):
        for section in meta["details"]:
            st.markdown(f"**{section['title']}**")
            st.markdown("\n".join(f"- {a}: {b}" for a, b in section["rows"]))
        c = meta.get("curve")
        if c:
            fig = go.Figure(go.Scatter(x=c["x"], y=c["y"], mode="lines+markers" if len(c["x"]) < 40 else "lines", line=dict(color=ACCENT)))
            fig.update_layout(title=dict(text=c["title"], font=dict(size=14)), height=260, margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_title=c["xlabel"], yaxis_title=c["ylabel"])
            st.plotly_chart(fig, width="stretch")

    if res is not None and meta["acts"]:
        if st.toggle("Draw the network for this sentence", key="network"):
            st.caption("Circles: the units that carried most of this answer, traced back from the outputs (twelve inputs, ten units per hidden layer), "
                       "highest value at the top, purple positive and red negative. Lines: the three strongest weight x value contributions into each "
                       "circle, purple pushing it up and red pushing it down, thicker for larger. The bar under each column is every unit of that layer "
                       "for this sentence, sorted from lowest to highest. Hover for the numbers.")
            st.plotly_chart(network_figure(meta, model, res), width="stretch")

    if res is not None and meta["encoder"]:
        if st.toggle("Show the [CLS] values", key="cls_values"):
            half = len(res["x"]) // 2
            cls = res["x"][:half]
            st.caption(f"The first {half:,} of the encoder's {len(res['x']):,} values: the vector bge-large gives at its [CLS] token, its summary of the whole text "
                       f"(averaged by length when a long text is read in several windows). Range {cls.min():.3f} to {cls.max():.3f}, length {np.linalg.norm(cls):.2f}.")
            table({"position": np.arange(half), "value": np.round(cls, 5)}, rows=12)
