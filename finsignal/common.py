import hashlib, html, re, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'config.json').read_text())
def clean(text):
    return re.sub(r'\s+',' ',re.sub(r'https?://\S+','',html.unescape(str(text)))).strip()
def key(text):
    return hashlib.sha256(clean(text).casefold().encode()).hexdigest()
