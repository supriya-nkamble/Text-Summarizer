# Text-Summarizer

A config-driven MLOps pipeline for abstractive dialogue summarization on the
[SAMSum](https://huggingface.co/datasets/samsum) corpus.

## Model & results

The production model is `facebook/bart-large-xsum` fine-tuned on SAMSum for 3
epochs. That base checkpoint was chosen deliberately, not by default: SAMSum's
reference summaries are short and highly abstractive, which matches XSum's
summary style far better than CNN/DailyMail's longer, more extractive style —
confirmed empirically below, not just asserted.

| model | base checkpoint's prior fine-tune | rouge1 | rouge2 | rougeL | bertscore-f1 |
|---|---|---|---|---|---|
| original baseline | `pegasus-cnn_dailymail` (news, extractive-leaning) | 43.6 | 21.0 | 34.9 | 0.849 |
| published reference (`lidiya/bart-large-xsum-samsum`) | `bart-large-xsum` (news, abstractive) | 51.6 | 27.5 | 42.8 | 0.869 |
| **this repo's model** | `bart-large-xsum`, fine-tuned here | **51.9** | **28.0** | **43.3** | **0.870** |

Full evaluation methodology, benchmark run, and training run live under
`kaggle/` (run on a Kaggle GPU kernel — local hardware here is an 8GB Apple M2
with no CUDA, impractical for beam-search generation over the full 819-example
test set). BERTScore uses `distilbert-base-uncased` unrescaled.

**Data quality note**: 41 of the 819 test examples (~5%) have dialogue text
byte-identical to an entry in the 14,732-example training set — likely
duplicate conversations in the original SAMSum corpus, not a split bug (there
is zero `id` overlap between splits). Checked whether this inflates the
reported numbers (`kaggle/check-leakage/`): those 41 examples do score higher
(ROUGE-1 56.3 vs 51.7 BERTScore F1 0.883 vs 0.869), but excluding them entirely
barely moves the aggregate (51.7/27.7/43.1, BERTScore F1 0.869 vs the
51.9/28.0/43.3, 0.870 reported above) since they're only 5% of the set.

## Pipeline

Each stage is a config-driven component, orchestrated by `main.py`:

1. **Data ingestion** — fetch and unpack the SAMSum conversational-summary dataset
2. **Data validation** — schema / file checks
3. **Data transformation** — tokenize with the model tokenizer
4. **Model training** — fine-tune a Transformer summarizer via the `transformers` Trainer
5. **Model evaluation** — ROUGE + BERTScore on the held-out split (`scripts/run_evaluation.py` re-runs this stage alone)
6. **Prediction** — served behind a FastAPI endpoint

Configuration lives in `config/config.yaml` + `params.yaml`; stage inputs/outputs
are typed entities; artifacts land under `artifacts/`.

## Run

```bash
git clone https://github.com/supriya-nkamble/Text-Summarizer
cd Text-Summarizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python main.py          # run the full pipeline
python app.py           # serve the FastAPI app on localhost
```

## Engineering

- `src/text_summarizer/` package (components / pipeline stages / config / entity / utils)
- `pytest` suite (data validation, prediction pipeline, API)
- GitHub Actions CI, `Dockerfile`, `ruff`
- Research notebooks for each stage under `research/`

## Attribution

The end-to-end structure follows a well-known tutorial pattern; the packaging,
tests, CI, and container work here are my own hardening on top of it.

## License

MIT
