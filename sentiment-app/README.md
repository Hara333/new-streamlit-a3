# Sentiment prediction

A Streamlit app that reads a sentence with any of the fourteen models from the COS60011 sentiment project. Pick a model from the dropdown, type a sentence and press Predict.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

The first sentence read by a network or a linear model downloads the encoder, BAAI/bge-large-en-v1.5 (about 1.3 GB), into the Hugging Face cache. The word logits never load it. Everything runs on the CPU. Tested with Python 3.13, Streamlit 1.66, plotly 7.1, numpy 2.5, torch 2.14 and transformers 5.18. A machine without internet access can point the app at a local copy of the encoder: `SENTIMENT_ENCODER=/path/to/bge-large-en-v1.5 streamlit run app.py`.

## Models

The method is in brackets after each name, as in the dropdown.

| Model | Trained on | Reads |
|---|---|---|
| FinGPT Network [deep learning] | FinGPT sentiment training set, 25,892 development rows | sentiment (3) |
| FinGPT Network + SEntFiN, all rows [deep learning, reinforced: closed form] | FinGPT Network, then all 5,266 headlines of the SEntFiN training half | sentiment |
| FinGPT Network + SEntFiN, mistakes only [deep learning, reinforced: closed form] | FinGPT Network, then the 1,399 training-half headlines it read wrongly | sentiment |
| FinGPT Network + SEntFiN, GRPO [deep learning, reinforced: GRPO] | FinGPT Network, then all 5,266 training-half headlines | sentiment |
| Dialogue Network [deep learning] | EmpatheticDialogues, 55,341 development rows with sentiment and emotion labels | sentiment (3) and emotion (32) |
| Dialogue Network + FinGPT [deep learning, reinforced: GRPO] | Dialogue Network, then 25,892 FinGPT texts | sentiment and emotion |
| Dialogue Network + FinGPT + SEntFiN, mistakes only [deep learning, reinforced twice: GRPO, then closed form] | Dialogue Network + FinGPT, then the 1,351 training-half headlines it read wrongly | sentiment and emotion |
| Dialogue Network + FinGPT + SEntFiN, all rows [deep learning, reinforced twice: GRPO, then closed form] | Dialogue Network + FinGPT, then all 5,266 training-half headlines | sentiment and emotion |
| Dialogue Network + FinGPT + SEntFiN, GRPO [deep learning, reinforced twice: GRPO, then GRPO] | Dialogue Network + FinGPT, then all 5,266 training-half headlines | sentiment and emotion |
| FinGPT Linear [linear] | the FinGPT Network's rows and folds | sentiment |
| Dialogue Linear [linear] | the Dialogue Network's rows and folds | sentiment and emotion |
| Word Logit, FinGPT [MNLM] | the FinGPT Network's rows, nine shades kept | nine shades, and the polarity they sum to |
| Word Logit, SEntFiN [MNLM] | SEntFiN, 9,027 development headlines | sentiment |
| Word Logit, Dialogue [MNLM] | the Dialogue Network's rows | emotion (32) |

The networks and linear heads read the 2,048 values of the frozen encoder (the [CLS] vector and the mean over the tokens). The word logits read the words themselves: a word never seen in training is N/A and adds nothing, and a sentence with no known word gets no reading.

## The page

- **Result**: the model's reading, in the form that model gives it (sentiment, emotion, shades, or a table of words for the word logits).
- **Show the calculation**: the arithmetic for this sentence. For the networks: the encoder values, standardisation, every hidden layer, the output logits, the temperature and the softmax. For a linear head the same without hidden layers; for a word logit, the intercepts plus each known word's coefficients.
- **Model details**: data, how the model was chosen and trained (stochastic gradient descent settings where there was any; the linear heads and word logits were fitted by L-BFGS), accuracy and macro-F1 on held-out rows, run time and memory, and a training curve.
- **Draw the network for this sentence** (deep learning models only, at the bottom): the units that carried most of the answer and the strongest contributions between them. Nothing is computed until it is switched on.
- **Show the [CLS] values** (models that use the encoder, last on the page): the 1,024 values of the [CLS] vector for this sentence.

## Power BI tab

The second tab shows a Power BI report inside the app. Paste the report's link in the tab, or start the app with it set so it loads on its own:

```
POWERBI_URL="https://app.powerbi.com/view?r=..." streamlit run app.py
```

In Power BI, File > Embed report > Publish to web gives a public link; Website or portal gives a secure one that asks viewers to sign in. Only links that start with https://app.powerbi.com/ are shown.

## Files

```
app.py                  the app
build_models.py         writes models/ from the pipeline's result files
models/catalog.json     names, readings and the facts the details panel shows
models/<model>.npz      the weights, one file per model
requirements.txt
requirements-test.txt   what the tests need on top
.streamlit/config.toml  theme colour
tests/                  the test suite (tests/README.md says how to run it)
```

## Tests

`pip install -r requirements-test.txt` then `pytest`. The suite covers the text path, the arithmetic, the model store, the readings against the pipeline's own answers, the page under Streamlit's AppTest, and the non-functional budgets; tests/README.md lists the files and the identifiers used in the test plan.

The weights are plain arrays loaded with `allow_pickle=False`; nothing is unpickled.

## Rebuilding models/

models/ is already built. After a new run of the pipeline:

```
python build_models.py --results <results folder> --logs <job logs folder> --out models
```

`--results` is laid out as the pipeline writes it:

```
fd/train/nn/            nn_model.json, nn_selected.json
fd/train/linear/        linear_model.json, linear_summary.json
fd/sentfin/             prediction_metrics.csv, linear/metrics.json
fd/rl/modelA_sentfin_mistakes/{exact_all,exact_wrong,grpo_all}/    nn_model.json, rl_summary.json, rl_log.csv
ed/train/nn/            nn_model.json, nn_selected.json
ed/train/linear/        linear_model.json, linear_summary.json
ed/sentfin/             prediction_summary.json, linear/metrics.json
ed/rl/suite_standard_train/grpo_full/                              nn_model.json, rl_summary.json, rl_log.csv
ed/rl/ed_fingpt_then_sentfin_mistakes/{exact_all,exact_wrong,grpo_all}/    the same three files
bow/{fingpt,sentfin,empathetic}/    bow_model.json, bow_summary.json
```

`--logs` is optional. It holds the job logs (`fd_nn_*.txt`, `ed_nn_*.txt`) whose closing report gives each network's run time and peak memory.

## Differences from the pipeline

The encoder runs here in float32 on the CPU. The training features were computed in float16 on a GPU and stored in float16, so a probability can differ slightly from the pipeline's own prediction for the same text. Fed the same encoder values (47 stored from real texts), the networks and linear heads here give the pipeline's probabilities to within 6e-7 and the same label every time; the word logits match exactly.
