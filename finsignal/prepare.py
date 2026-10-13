"""Download exact source revisions; prepare annotation queue and sentiment splits."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download
from sklearn.model_selection import train_test_split
from .common import clean,key
def main():
    a=argparse.ArgumentParser(); a.add_argument('--out',default='data'); a.add_argument('--snapshot',help='Directory containing downloaded raw CSVs and source_manifest.json'); args=a.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); manifest={}
    snapshot=Path(args.snapshot) if args.snapshot else None
    locked=json.loads((snapshot/'source_manifest.json').read_text()) if snapshot else json.loads((Path(__file__).resolve().parents[1]/'source_lock.json').read_text())
    for task in ['topic','sentiment']:
        repo='zeroshot/twitter-financial-news-'+task
        rev=locked[task]['revision']; manifest[task]={'repo':repo,'revision':rev,'license':'MIT'}
        tables=[]
        for split in ['train','valid']:
            name=f'{"sent" if task=="sentiment" else task}_{split}.csv'
            p=snapshot/name if snapshot else hf_hub_download(repo,name,repo_type='dataset',revision=rev)
            expected=locked[task].get('files',{}).get(name,{}).get('sha256')
            if expected and hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected:raise ValueError('Source snapshot hash mismatch')
            df=pd.read_csv(p); df['text']=df.text.map(clean); df['id']=df.text.map(key)
            df['source_split']=split; df['source_repo']=repo; df['source_revision']=rev
            tables.append(df)
        df=pd.concat(tables,ignore_index=True); df=df[df.text.str.len()>0]
        conflicts=df.groupby('id').label.nunique(); df=df[~df.id.isin(conflicts[conflicts>1].index)]
        df=df.drop_duplicates('id'); df.to_csv(out/f'{task}_source.csv',index=False)
        if task=='topic':
            # Balance acquisition across original topics, NOT assumed event labels.
            queue=pd.concat([g.sample(min(100,len(g)),random_state=42) for _,g in df.groupby('label')],ignore_index=True).rename(columns={'label':'source_topic'})
            for col in ['annotator_1','annotator_2','event_label','story_group','review_status','notes']: queue[col]=''
            queue.to_csv(out/'event_annotation.csv',index=False)
        else:
            df['label']=df.label.map({0:'negative',1:'positive',2:'neutral'})
            test=df[df.source_split=='valid']; pool=df[df.source_split=='train']
            train,val=train_test_split(pool,test_size=.15,stratify=pool.label,random_state=42)
            for split,t in [('train',train),('validation',val),('test',test)]:t.to_csv(out/f'sentiment_{split}.csv',index=False)
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
