import json
from pathlib import Path
import pandas as pd
import streamlit as st
from finsignal.inference import analyze
st.set_page_config(page_title='FinSignal | Group 7',page_icon='📰',layout='wide')
st.title('FinSignal')
st.caption('Financial news intelligence • Wealthsimple research prototype • ISOM5240 Group 7')
st.write('Turn company news into sentiment and corporate event labels for faster research.')
st.info('Research decision support. Confidence is a model score, not a probability of investment success. Long articles are truncated to 256 tokens.')
p=Path(__file__).with_name('deployment_manifest.json')
if not p.exists():
    st.warning('Event model training and selection are pending. Add the selected model manifest to enable analysis.');st.stop()
manifest=json.loads(p.read_text());single,batch=st.tabs(['Single news item','Batch dashboard'])
def show(texts):
    if not texts or any(not str(t).strip() for t in texts):st.error('Enter non-empty news text.');return
    if len(texts)>500:st.error('Please use at most 500 rows per batch.');return
    try:
        with st.spinner('Analyzing news…'):df=pd.DataFrame(analyze(texts,manifest))
        st.dataframe(df,use_container_width=True);a,b=st.columns(2)
        a.bar_chart(df.sentiment.value_counts());b.bar_chart(df.event.value_counts())
        st.download_button('Download results',df.to_csv(index=False),'finsignal_results.csv','text/csv')
    except Exception as exc:st.error(f'Analysis could not complete: {exc}')
with single:
    text=st.text_area('Headline or article',max_chars=10000)
    if st.button('Analyze news'):show([text])
with batch:
    uploaded=st.file_uploader('CSV with a text column',type=['csv'])
    if uploaded and st.button('Analyze batch'):
        try:
            frame=pd.read_csv(uploaded)
            if 'text' not in frame or frame.text.isna().any():st.error('CSV requires a text column without missing values.')
            else:show(frame.text.astype(str).tolist())
        except Exception as exc:st.error(f'Cannot read CSV: {exc}')
