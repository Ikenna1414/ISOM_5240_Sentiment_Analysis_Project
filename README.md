# FinSignal

**ISOM5240 Group 7** - SHI Yuebing and Ikenna Odikamnoro

A transformer-based financial news intelligence prototype for the Wealthsimple business case. FinSignal produces sentiment, corporate event labels and confidence scores for single news items and CSV batches.

## Current status

Project implementation and verified dataset snapshots are prepared. Event annotation review, transformer training, measured model comparison and public deployment are pending. The app shows a training-pending message until selected models are configured.

## Two pipelines

- Sentiment: positive, neutral or negative. Train DistilBERT and RoBERTa candidates; compare with unchanged ProsusAI/finbert.
- Corporate events: earnings/guidance, mergers/acquisitions, regulatory/legal, financing, management change, product/business expansion or other. Fine-tune DistilBERT and RoBERTa after reviewing event labels.

Select on validation accuracy plus CPU runtime. Evaluate the selected application models on frozen held-out tests. Model scores are uncalibrated confidence, not investment advice.

## Repository contents

```text
app.py                         Streamlit entry point
requirements.txt               App dependencies
requirements-training.txt      Additional experiment dependencies
config.json                    Label schema and training settings
source_lock.json               Immutable data/model source revisions
deployment_manifest.example.json
finsignal/                     Preparation, training, inference and evaluation
notebooks/                     Eight reproducible notebooks
docs/                          Dataset decisions and checkpoint evidence
examples/news_batch.csv        Illustrative input only
results/                       Measured experiment outputs when available
ANNOTATION_GUIDE.md             Seven-event review protocol
MODEL_CARD_TEMPLATE.md          Complete before publishing models
```

## Run locally

Use Python 3.11 and a virtual environment. From the repository root:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows, or `source .venv/bin/activate` on macOS/Linux, then:

```text
python -m pip install -r requirements.txt
streamlit run app.py
```

Until model selection is complete, the app displays its pending state. To enable predictions, create deployment_manifest.json containing the selected Hugging Face model IDs and immutable commit revisions. See the example manifest. The event model must use the seven labels in config.json.

## Reproduce experiments

```text
python -m pip install -r requirements-training.txt
python -m finsignal.prepare --out data
```

Review data/event_annotation.csv according to ANNOTATION_GUIDE.md. Source topics are not event labels. Keep approved rows, resolve disagreements and assign related-story groups before:

```text
python -m finsignal.split_events data/event_annotation.csv --out data
python -m finsignal.train --task event --model distilbert/distilbert-base-uncased --seed 42 --out runs/event_distilbert_42
```

Run seeds 42, 43 and 44 for each of DistilBERT and RoBERTa on both tasks. Run the unchanged sentiment reference:

```text
python -m finsignal.train --task sentiment --model ProsusAI/finbert --reference --out runs/sentiment_finbert
python -m finsignal.export_results --out results/Experimental_results.xlsx
python -m finsignal.select
python -m finsignal.evaluate --out results/held_out_accuracy.json
```

The eight notebooks cover source preparation, reviewed event splitting, all candidates, selection and final evaluation. Start them with notebooks/ as the working directory. Model selection uses highest mean validation accuracy; within 0.01 absolute accuracy choose lower CPU latency. Deployment uses the selected candidate's predeclared seed-42 checkpoint. Test accuracy is not used for selection.

## Data provenance

Sources: [Twitter Financial News sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) and [Twitter Financial News topics](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic). Source cards declare MIT; retain their notices and attribution. source_lock.json pins revisions and hashes.

Topic annotations do not directly represent the proposal's seven events. The project uses the proposal-permitted fallback of a reviewed project-specific subset of public financial text. The submission package includes an unlabelled review queue of 1,922 items. FinBERT was trained on Financial PhraseBank; do not use PhraseBank as an independent test for it. The current FinBERT card declares CC-BY-NC-SA-4.0. This academic prototype does not establish commercial licensing clearance.

## Hugging Face and Streamlit

After selection, complete each model card and publish selected models/tokenizers to Hugging Face. The publish helper uses HF_TOKEN from the environment and records publication commits. Never commit credentials.

For Streamlit Community Cloud, use this repository's root app.py and Python 3.11. See docs/GITHUB_SETUP.md for the upload and deployment sequence. Commit deployment_manifest.json only after replacing local model paths with published model IDs/revisions and removing that filename from .gitignore.

## Limitations

Source texts are short financial tweets, not a validated benchmark for Canadian news or long articles. Input truncates to 256 tokens. Exact-text deduplication has passed; near-duplicate and related-story review remains necessary. No measured accuracy, winning model, live deployment or investment benefit is claimed yet. This is an academic prototype, with no Wealthsimple affiliation or endorsement.
