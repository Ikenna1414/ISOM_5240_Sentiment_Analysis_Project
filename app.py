"""
FinSignal - Transformer-Based Financial News Intelligence for Wealthsimple
ISOM5240 Group 7 (SHI Yuebing, Ikenna Odikamnoro)

Two Hugging Face pipelines, both fine-tuned by the team:
  Pipeline 1: text-classification -> market sentiment (Bearish / Bullish / Neutral)
  Pipeline 2: text-classification -> corporate event type (7 categories from the proposal)
"""
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

import utils as U

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="FinSignal | Financial News Intelligence",
                   page_icon="📈", layout="wide")


# ---------------------------------------------------------------- models ----
@st.cache_resource(show_spinner="Loading fine-tuned models from Hugging Face...")
def load_pipelines():
    sent = pipeline("text-classification", model=U.SENTIMENT_MODEL_ID, device=-1)
    event = pipeline("text-classification", model=U.EVENT_MODEL_ID, device=-1)
    return sent, event


def run_pipeline(pipe, texts, batch_size=32):
    """Return list of full score distributions, one per input text."""
    out = pipe(list(texts), batch_size=batch_size, top_k=None,
               truncation=True, max_length=U.MAX_LEN)
    # A single input can come back un-nested; normalise to list-of-lists
    if out and isinstance(out[0], dict):
        out = [out]
    return out


def analyse(texts):
    """Run both pipelines on raw texts and return a tidy DataFrame."""
    sent_pipe, event_pipe = load_pipelines()
    cleaned = [U.clean_text(t) for t in texts]
    safe = [c if c else "(empty)" for c in cleaned]
    t0 = time.perf_counter()
    sent_scores = run_pipeline(sent_pipe, safe)
    event_scores = run_pipeline(event_pipe, safe)
    elapsed = time.perf_counter() - t0
    rows = []
    for raw, c, ss, es in zip(texts, cleaned, sent_scores, event_scores):
        s_lbl, s_conf = U.top_prediction(ss)
        e_lbl, e_conf = U.top_prediction(es)
        rows.append({
            "text": raw, "clean_text": c,
            "sentiment": s_lbl, "sentiment_conf": round(s_conf, 4),
            "event": e_lbl, "event_conf": round(e_conf, 4),
            "attention": U.attention_level(s_lbl, s_conf, e_lbl),
            "_sent_scores": ss, "_event_scores": es,
        })
    return pd.DataFrame(rows), elapsed


def score_chart(scores, color_map=None, height=180):
    df = pd.DataFrame(scores).sort_values("score", ascending=False)
    enc_color = (alt.Color("label:N", scale=alt.Scale(domain=list(color_map),
                 range=list(color_map.values())), legend=None)
                 if color_map else alt.value("#2b6cb0"))
    return (alt.Chart(df).mark_bar().encode(
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
        y=alt.Y("label:N", sort="-x", title=None),
        color=enc_color,
        tooltip=["label", alt.Tooltip("score:Q", format=".1%")],
    ).properties(height=height))


# ---------------------------------------------------------------- header ----
st.title("📈 FinSignal")
st.markdown(
    "**Transformer-based financial news intelligence for Wealthsimple.** "
    "Paste a headline or upload a batch of news. FinSignal returns the market "
    "**sentiment** and the **corporate event type** of every item, with confidence "
    "scores, a triage flag and a batch summary."
)
st.caption("Decision-support tool for research triage. Not investment advice.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Pre-processing**: HTML unescape, URL removal, whitespace clean-up\n"
        "2. **Pipeline 1 - Sentiment**: fine-tuned DistilBERT\n"
        "3. **Pipeline 2 - Event type**: fine-tuned DistilBERT (7 categories)\n"
        "4. **Post-processing**: attention flag + batch summary"
    )
    st.markdown("**Models (Hugging Face)**")
    st.code(f"{U.SENTIMENT_MODEL_ID}\n{U.EVENT_MODEL_ID}", language=None)
    with st.expander("Event categories"):
        for k, v in U.EVENT_DESCRIPTIONS.items():
            st.markdown(f"**{k}**: {v}")
    with st.expander("Attention flag rules"):
        st.markdown(
            "- **High**: Bearish, high-impact event, sentiment confidence >= 70%\n"
            "- **Medium**: any other Bearish item, or Bullish + high-impact event\n"
            "- **Low**: everything else\n\n"
            "High-impact events: " + ", ".join(sorted(U.HIGH_IMPACT_EVENTS))
        )
    st.caption("ISOM5240 Group 7 · SHI Yuebing · Ikenna Odikamnoro")

tab_single, tab_batch, tab_eval = st.tabs(
    ["🔎 Single headline", "📊 Batch dashboard", "✅ Evaluate accuracy"])

EXAMPLES = [
    "Shares slide after the retailer cuts its full-year profit outlook",
    "Chipmaker agrees to buy rival software firm in $8 billion deal",
    "Regulators open probe into bank's overdraft fee practices",
    "Automaker names new chief executive as founder steps down",
    "Software maker beats quarterly estimates and raises annual guidance",
    "Mining company prices $2 billion bond offering to fund expansion",
]

# ------------------------------------------------------------ single tab ----
with tab_single:
    choice = st.selectbox("Try an example or write your own below",
                          ["(write my own)"] + EXAMPLES)
    default = "" if choice == "(write my own)" else choice
    text = st.text_area("News headline or short article", value=default, height=110,
                        placeholder="e.g. Company X beats Q3 estimates and raises guidance")
    if st.button("Analyse", type="primary", disabled=not text.strip()):
        df, secs = analyse([text])
        r = df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", r.sentiment, f"{r.sentiment_conf:.0%} confidence",
                  delta_color="off")
        c2.metric("Event type", r.event, f"{r.event_conf:.0%} confidence",
                  delta_color="off")
        c3.metric("Attention", r.attention, f"{secs*1000:.0f} ms", delta_color="off")
        st.info(U.interpret(r.sentiment, r.event))
        g1, g2 = st.columns(2)
        g1.markdown("**Sentiment probabilities**")
        g1.altair_chart(score_chart(r._sent_scores, U.SENTIMENT_COLORS, 130),
                        use_container_width=True)
        g2.markdown("**Event-type probabilities**")
        g2.altair_chart(score_chart(r._event_scores, None, 220),
                        use_container_width=True)
        with st.expander("Pre-processed text sent to the models"):
            st.write(r.clean_text or "(empty after cleaning)")

# ------------------------------------------------------------- batch tab ----
with tab_batch:
    st.markdown("Upload a CSV with a text column, or paste one headline per line.")
    src = st.radio("Input", ["Upload CSV", "Paste headlines", "Use bundled sample"],
                   horizontal=True)
    texts = []
    if src == "Upload CSV":
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            col = st.selectbox("Text column", raw.columns,
                               index=list(raw.columns).index("text")
                               if "text" in raw.columns else 0)
            texts = raw[col].astype(str).tolist()
    elif src == "Paste headlines":
        pasted = st.text_area("One headline per line", height=180)
        texts = [t for t in pasted.splitlines() if t.strip()]
    else:
        sample = DATA_DIR / "sample_news.csv"
        if sample.exists():
            texts = pd.read_csv(sample)["text"].astype(str).tolist()
            st.caption(f"{len(texts)} sample headlines loaded from data/sample_news.csv")

    if len(texts) > U.MAX_BATCH_ROWS:
        st.warning(f"Only the first {U.MAX_BATCH_ROWS} rows will be analysed.")
        texts = texts[: U.MAX_BATCH_ROWS]

    if st.button("Run batch analysis", type="primary", disabled=not texts):
        with st.spinner(f"Analysing {len(texts)} items..."):
            df, secs = analyse(texts)
        st.session_state["batch_df"], st.session_state["batch_secs"] = df, secs

    if "batch_df" in st.session_state:
        df, secs = st.session_state["batch_df"], st.session_state["batch_secs"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Items", len(df))
        m2.metric("Bullish", f"{(df.sentiment == 'Bullish').mean():.0%}")
        m3.metric("Bearish", f"{(df.sentiment == 'Bearish').mean():.0%}")
        m4.metric("High attention", int((df.attention == "High").sum()))
        st.caption(f"Processed in {secs:.1f} s ({secs / len(df) * 1000:.0f} ms per item, CPU)")

        st.subheader("Summary")
        for line in U.build_summary(df):
            st.markdown(f"- {line}")

        ch1, ch2 = st.columns(2)
        sc = df.sentiment.value_counts().rename_axis("sentiment").reset_index(name="count")
        ch1.altair_chart(alt.Chart(sc, title="Sentiment mix").mark_arc(innerRadius=50).encode(
            theta="count:Q",
            color=alt.Color("sentiment:N", scale=alt.Scale(
                domain=list(U.SENTIMENT_COLORS), range=list(U.SENTIMENT_COLORS.values()))),
            tooltip=["sentiment", "count"]), use_container_width=True)
        ec = df.groupby(["event", "sentiment"]).size().reset_index(name="count")
        ch2.altair_chart(alt.Chart(ec, title="Event type by sentiment").mark_bar().encode(
            x=alt.X("count:Q"), y=alt.Y("event:N", sort="-x", title=None),
            color=alt.Color("sentiment:N", scale=alt.Scale(
                domain=list(U.SENTIMENT_COLORS), range=list(U.SENTIMENT_COLORS.values()))),
            tooltip=["event", "sentiment", "count"]), use_container_width=True)

        st.subheader("Priority queue")
        order = {"High": 0, "Medium": 1, "Low": 2}
        view = (df.assign(_o=df.attention.map(order))
                  .sort_values(["_o", "sentiment_conf"], ascending=[True, False])
                  [["attention", "sentiment", "sentiment_conf", "event", "event_conf", "text"]])
        flt = st.multiselect("Show attention levels", ["High", "Medium", "Low"],
                             default=["High", "Medium", "Low"])
        st.dataframe(view[view.attention.isin(flt)], use_container_width=True, hide_index=True,
                     column_config={
                         "sentiment_conf": st.column_config.ProgressColumn(
                             "sent. conf", min_value=0, max_value=1, format="%.2f"),
                         "event_conf": st.column_config.ProgressColumn(
                             "event conf", min_value=0, max_value=1, format="%.2f")})
        st.download_button("Download results (CSV)",
                           view.to_csv(index=False).encode("utf-8"),
                           "finsignal_results.csv", "text/csv")

# -------------------------------------------------------------- eval tab ----
with tab_eval:
    st.markdown(
        "Measure the deployed app's **accuracy** on labelled, held-out test data. "
        "Use the bundled test sets (exported by the testing notebook) or upload your own "
        "CSV with a `text` column and a label column."
    )
    which = st.radio("Pipeline to evaluate", ["Pipeline 1 - Sentiment", "Pipeline 2 - Event type"],
                     horizontal=True)
    is_sent = which.startswith("Pipeline 1")
    labels = U.SENTIMENT_LABELS if is_sent else U.EVENT_LABELS
    bundled = DATA_DIR / ("app_test_sentiment.csv" if is_sent else "app_test_event.csv")

    eval_src = st.radio("Test data", ["Bundled test set", "Upload labelled CSV"], horizontal=True)
    eval_df = None
    if eval_src == "Bundled test set":
        if bundled.exists():
            eval_df = pd.read_csv(bundled)
            st.caption(f"{len(eval_df)} labelled samples from data/{bundled.name}")
        else:
            st.warning(f"data/{bundled.name} not found in the repository.")
    else:
        up = st.file_uploader("Labelled CSV", type=["csv"], key="eval_up")
        if up is not None:
            eval_df = pd.read_csv(up)

    if eval_df is not None:
        cols = list(eval_df.columns)
        tcol = st.selectbox("Text column", cols, index=cols.index("text") if "text" in cols else 0)
        lcol = st.selectbox("Label column", cols, index=cols.index("label") if "label" in cols
                            else min(1, len(cols) - 1))
        if st.button("Run evaluation", type="primary"):
            data = eval_df[[tcol, lcol]].copy()
            data["y_true"] = data[lcol].map(lambda v: U.normalise_label(v, labels))
            skipped = int(data.y_true.isna().sum())
            data = data.dropna(subset=["y_true"]).head(U.MAX_BATCH_ROWS)
            sent_pipe, event_pipe = load_pipelines()
            pipe = sent_pipe if is_sent else event_pipe
            texts = [U.clean_text(t) or "(empty)" for t in data[tcol]]
            with st.spinner(f"Scoring {len(texts)} samples on Streamlit Cloud CPU..."):
                t0 = time.perf_counter()
                preds = [U.top_prediction(s)[0] for s in run_pipeline(pipe, texts)]
                secs = time.perf_counter() - t0
            data["y_pred"] = preds
            acc, mf1, table = U.classification_table(data.y_true, data.y_pred, labels)
            e1, e2, e3 = st.columns(3)
            e1.metric("Accuracy", f"{acc:.2%}")
            e2.metric("Samples", len(data))
            e3.metric("Runtime", f"{secs:.1f} s")
            st.caption(f"Macro-F1 (supplementary): {mf1:.3f}"
                       + (f" · {skipped} rows skipped (unknown label)" if skipped else ""))
            st.markdown("**Per-class results**")
            st.dataframe(table.round(3), use_container_width=True, hide_index=True)
            st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
            st.dataframe(U.confusion(data.y_true, data.y_pred, labels),
                         use_container_width=True)
            st.download_button("Download predictions (CSV)",
                               data.to_csv(index=False).encode("utf-8"),
                               "finsignal_eval_predictions.csv", "text/csv")
