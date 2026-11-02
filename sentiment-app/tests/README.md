# Tests of the Streamlit app

```
pip install -r requirements-test.txt
pytest                      # everything but the load test and the real encoder: about a minute
pytest -m "not slow"        # the quick checks only: a few seconds
pytest tests/test_load.py   # the load test: starts a server and drives it with Playwright
pytest --real-encoder tests/test_encode.py   # one check with the real bge-large (downloads it)
COS_RESULTS=<results folder> COS_LOGS=<logs folder> pytest tests/test_build_models.py   # rebuild models/ and compare
```

`playwright install chromium` is needed once for the load test.

| File | Level | What it checks | IDs |
|---|---|---|---|
| test_text.py | component | cleaning and tokenising of the typed sentence, against the pipeline's cleaning (golden/cleaning.json) | ST-T |
| test_maths.py | component | softmax, activations, the forward pass, the word logit sum, percentage formatting, drawing helpers; synthetic weights with closed-form answers | ST-M |
| test_models.py | component | models/: catalog entries, npz layout, weight counts, vocabulary rules, no pickle, size, house style | ST-F |
| test_parity.py | component and regression | every model's readings against the pipeline's own answers on 47 real encoder vectors and twelve sentences (golden/) | ST-P |
| test_encode.py | component | windowing of long texts at the 510/511 and 988/989 token boundaries, token-weighted average, cache hits; a stub stands in for the encoder | ST-E |
| test_ui.py | system | the page under Streamlit's AppTest: every model, every switch, unknown words, hostile input, the encoder missing | ST-U |
| test_nonfunctional.py | system | response time, load time, memory, size, security scan, lint, complexity, minimum Python, theme | NF |
| test_load.py | system | one and several concurrent browsers against a running server | NF-L |
| test_build_models.py | integration | build_models.py from the pipeline's results reproduces the shipped store byte for byte | ST-B |

The stand-in encoder is a one-layer BERT with bge-large's output width and random weights, built once
per session; it makes the readings meaningless and the flow testable without the 1.3 GB download.
The app reads the encoder's location from `SENTIMENT_ENCODER` when that variable is set, which is
how the tests point it at the stand-in (and how a deployment can point it at a local copy).

`tests/tools/make_golden.py` rewrites the golden files from the pipeline's code; run it after a new
training run, then the parity tests describe the new models.
