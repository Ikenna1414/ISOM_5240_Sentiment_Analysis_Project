"""
FinSignal - shared helpers for the Streamlit app.

The text-cleaning function and the label sets below are IDENTICAL to the ones
used in the Colab fine-tuning notebooks, so the deployed app sees text in
exactly the same form the models were trained on.
"""
import html
import os
import re

import pandas as pd

# --------------------------------------------------------------------------
# Model configuration (fine-tuned models published on Hugging Face)
# --------------------------------------------------------------------------
HF_USERNAME = os.environ.get("HF_USERNAME", "Ikenna14")  # Hugging Face account hosting the fine-tuned models

SENTIMENT_MODEL_ID = os.environ.get(
    "SENTIMENT_MODEL_ID", f"{HF_USERNAME}/finsignal-sentiment-distilbert"
)
EVENT_MODEL_ID = os.environ.get(
    "EVENT_MODEL_ID", f"{HF_USERNAME}/finsignal-event-distilbert"
)

MAX_LEN = 96          # same max sequence length as fine-tuning
MAX_BATCH_ROWS = 1000  # safety cap for Streamlit Community Cloud (CPU)

# --------------------------------------------------------------------------
# Label sets (must match id2label saved in the fine-tuned model configs)
# --------------------------------------------------------------------------
SENTIMENT_LABELS = ["Bearish", "Bullish", "Neutral"]

EVENT_LABELS = [
    "Earnings & Guidance",
    "Mergers & Acquisitions",
    "Regulatory & Legal",
    "Financing",
    "Management Change",
    "Product & Business Expansion",
    "Other",
]

EVENT_DESCRIPTIONS = {
    "Earnings & Guidance": "Quarterly results, revenue/EPS beats or misses, forward guidance.",
    "Mergers & Acquisitions": "Mergers, acquisitions, takeover bids, stake purchases.",
    "Regulatory & Legal": "Lawsuits, investigations, fines, regulatory approvals or actions.",
    "Financing": "Corporate debt issuance, IPOs and other capital raising.",
    "Management Change": "CEO/CFO/board appointments, departures and executive moves.",
    "Product & Business Expansion": "Product launches, contracts, partnerships, new markets or facilities.",
    "Other": "Macro, central banks, market moves, analyst calls, dividends and general commentary.",
}

# Events that typically move a single stock materially
HIGH_IMPACT_EVENTS = {
    "Earnings & Guidance",
    "Mergers & Acquisitions",
    "Regulatory & Legal",
    "Management Change",
}

SENTIMENT_COLORS = {"Bearish": "#d64545", "Bullish": "#1f9d55", "Neutral": "#8a8f98"}

# --------------------------------------------------------------------------
# Pre-processing (identical to the notebooks)
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Unescape HTML entities, remove URLs and normalise whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = html.unescape(str(text))
    text = _URL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# Post-processing / business logic
# --------------------------------------------------------------------------
def attention_level(sentiment: str, sent_conf: float, event: str) -> str:
    """Rule-based triage flag that turns two model outputs into an action cue."""
    high_impact = event in HIGH_IMPACT_EVENTS
    if sentiment == "Bearish" and high_impact and sent_conf >= 0.70:
        return "High"
    if sentiment == "Bearish" or (sentiment == "Bullish" and high_impact):
        return "Medium"
    return "Low"


def interpret(sentiment: str, event: str) -> str:
    """One-line plain-English reading of a (sentiment, event) pair."""
    tone = {
        "Bearish": "a negative signal",
        "Bullish": "a positive signal",
        "Neutral": "an informational item",
    }[sentiment]
    return f"{event} news carrying {tone}. {EVENT_DESCRIPTIONS.get(event, '')}"


def top_prediction(scores):
    """scores: list of {'label','score'} dicts -> (label, score)."""
    best = max(scores, key=lambda d: d["score"])
    return best["label"], float(best["score"])


def build_summary(df: pd.DataFrame) -> list:
    """Short auto-generated batch summary (bullet strings)."""
    n = len(df)
    if n == 0:
        return ["No items analysed."]
    s = df["sentiment"].value_counts(normalize=True)
    e = df["event"].value_counts()
    out = [
        f"{n} news items analysed: "
        f"{s.get('Bullish', 0):.0%} bullish, {s.get('Bearish', 0):.0%} bearish, "
        f"{s.get('Neutral', 0):.0%} neutral."
    ]
    top_events = ", ".join(f"{k} ({v})" for k, v in e.head(3).items())
    out.append(f"Most frequent event types: {top_events}.")
    high = int((df["attention"] == "High").sum())
    out.append(
        f"{high} item(s) flagged HIGH attention (bearish, high-impact event, confidence >= 70%)."
    )
    bearish_events = df[df["sentiment"] == "Bearish"]["event"].value_counts()
    if len(bearish_events):
        out.append(f"Bearish news is concentrated in: {bearish_events.index[0]}.")
    return out


# --------------------------------------------------------------------------
# Evaluation helpers (no scikit-learn needed on Streamlit Cloud)
# --------------------------------------------------------------------------
def normalise_label(value, valid_labels):
    """Map a ground-truth value (name, any case, or integer id) to a label name."""
    if pd.isna(value):
        return None
    lookup = {lbl.lower(): lbl for lbl in valid_labels}
    v = str(value).strip()
    if v.lower() in lookup:
        return lookup[v.lower()]
    if v.isdigit() and int(v) < len(valid_labels):
        return valid_labels[int(v)]
    return None


def classification_table(y_true, y_pred, labels):
    """Per-class precision / recall / F1 / support plus accuracy and macro-F1."""
    y_true = pd.Series(list(y_true), dtype="object")
    y_pred = pd.Series(list(y_pred), dtype="object")
    rows = []
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_true == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_true != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_true == lbl)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {"label": lbl, "precision": prec, "recall": rec, "f1": f1,
             "support": int((y_true == lbl).sum())}
        )
    table = pd.DataFrame(rows)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    present = table[table["support"] > 0]
    macro_f1 = float(present["f1"].mean()) if len(present) else 0.0
    return accuracy, macro_f1, table


def confusion(y_true, y_pred, labels):
    cm = pd.crosstab(
        pd.Categorical(list(y_true), categories=labels),
        pd.Categorical(list(y_pred), categories=labels),
        rownames=["actual"], colnames=["predicted"], dropna=False,
    )
    return cm
