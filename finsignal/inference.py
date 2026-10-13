from functools import lru_cache
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from .common import clean,CONFIG
@lru_cache(maxsize=4)
def load(model,revision=None):
    tok=AutoTokenizer.from_pretrained(model,revision=revision)
    net=AutoModelForSequenceClassification.from_pretrained(model,revision=revision).eval()
    return tok,net
def classify(model,texts,revision=None):
    tok,net=load(model,revision);result=[]
    for i in range(0,len(texts),16):
        batch=tok([clean(t) for t in texts[i:i+16]],padding=True,truncation=True,max_length=CONFIG['max_length'],return_tensors='pt')
        with torch.inference_mode():probs=net(**batch).logits.softmax(-1)
        for row in probs:
            k=int(row.argmax());result.append({'label':net.config.id2label[k].lower(),'confidence':float(row[k])})
    return result
def analyze(texts,manifest):
    s=classify(manifest['sentiment']['model'],texts,manifest['sentiment'].get('revision'))
    e=classify(manifest['event']['model'],texts,manifest['event'].get('revision'))
    if any(x['label'] not in CONFIG['labels'] for x in e):raise ValueError('Event model label schema does not match the proposal')
    return [{'text':t,'sentiment':a['label'],'sentiment_confidence':a['confidence'],'event':b['label'],'event_confidence':b['confidence']} for t,a,b in zip(texts,s,e)]
