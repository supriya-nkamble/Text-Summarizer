# Text-Summarizer

A config-driven MLOps pipeline for abstractive dialogue summarization. The point
of this repo is the **pipeline architecture and engineering practices**, not the
benchmark score — the checked-in demo model is trained for a single epoch, so its
ROUGE is not reported.

## Pipeline

Each stage is a config-driven component, orchestrated by `main.py`:

1. **Data ingestion** — fetch and unpack the SAMSum conversational-summary dataset
2. **Data validation** — schema / file checks
3. **Data transformation** — tokenize with the model tokenizer
4. **Model training** — fine-tune a Transformer summarizer via the `transformers` Trainer
5. **Model evaluation** — ROUGE on the held-out split
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
