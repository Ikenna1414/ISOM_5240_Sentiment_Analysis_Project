# FinSignal — Transformer-Based Financial News Intelligence

**ISOM5240 Group 7** · SHI Yuebing, Ikenna Odikamnoro
**Company:** Wealthsimple — https://www.wealthsimple.com/

FinSignal turns unstructured financial news into structured signals. For every headline it predicts:

| Pipeline | Task | Model (fine-tuned by the team) | Output |
|---|---|---|---|
| 1 | Market sentiment | `distilbert-base-uncased` → `finsignal-sentiment-distilbert` | Bearish / Bullish / Neutral + confidence |
| 2 | Corporate event type | `distilbert-base-uncased` → `finsignal-event-distilbert` | 7 event categories + confidence |

A rule-based post-processing step combines both outputs into an **attention flag** (High / Medium / Low)
and a batch **summary** for research triage.

```
headline ─► clean_text ─► Pipeline 1 (sentiment) ─┐
                     └──► Pipeline 2 (event)  ────┴─► attention flag + dashboard
```

## App features
- **Single headline** — sentiment, event type, probabilities and a plain-English interpretation
- **Batch dashboard** — CSV upload / paste, sentiment mix, event breakdown, priority queue, CSV download
- **Evaluate accuracy** — scores the deployed models on the bundled labelled test sets (`data/app_test_*.csv`)

## Repository layout
```
app.py, utils.py            Streamlit app (utils.py holds labels, cleaning, business rules)
requirements.txt            Pinned app dependencies (CPU PyTorch)
data/                       sample_news.csv, app_test_*.csv (held-out labelled samples),
                            event_mapping_spotcheck.xlsx (blind manual label check)
notebooks/                  01 sentiment fine-tuning, 02 event fine-tuning,
                            03 model selection, testing and Experimental_results.xlsx
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuration
Set your Hugging Face username in `utils.py` (`HF_USERNAME`) — the app loads
`<HF_USERNAME>/finsignal-sentiment-distilbert` and `<HF_USERNAME>/finsignal-event-distilbert`.

## Data
- Sentiment: [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- Event type: [zeroshot/twitter-financial-news-topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)
  (20 source topics mapped to the proposal's 7 event categories; mapping quality checked on a blind 150-headline manual sample)

*Decision-support tool for research triage — not investment advice.*
