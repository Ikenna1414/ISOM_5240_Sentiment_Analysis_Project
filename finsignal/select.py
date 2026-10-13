"""Select by validation only; produce local app manifest before HF publication."""
import argparse,json
from pathlib import Path
import pandas as pd
def main():
    a=argparse.ArgumentParser();a.add_argument('--runs',default='runs');a.add_argument('--out',default='deployment_manifest.json');args=a.parse_args()
    rows=[]
    for p in Path(args.runs).glob('*/results.json'):
        r=json.loads(p.read_text());r['model_path']=str((p.parent/'model').resolve());rows.append(r)
    manifest={};summary=[]
    for task in ['sentiment','event']:
        df=pd.DataFrame([r for r in rows if r['task']==task and r['stage']=='validation'])
        if df.empty:raise ValueError(f'Missing {task} results')
        expected_models={'distilbert/distilbert-base-uncased','FacebookAI/roberta-base'}
        if task=='sentiment':expected_models.add('ProsusAI/finbert')
        if set(df.model)!=expected_models:raise ValueError(f'Missing verified candidates for {task}')
        if df.split_sha256.map(lambda x:json.dumps(x,sort_keys=True)).nunique()!=1:raise ValueError('Candidates used different data splits')
        if df.platform.nunique()!=1:raise ValueError('Candidates used different platforms')
        if df.duplicated(['model','seed']).any():raise ValueError('Duplicate candidate/seed results')
        for model,g in df.groupby('model'):
            expected={42} if model=='ProsusAI/finbert' else {42,43,44}
            if set(g.seed)!=expected:raise ValueError(f'Incomplete seed runs: {model}')
        aggregate=df.groupby('model').agg(accuracy=('accuracy','mean'),accuracy_sd=('accuracy','std'),latency=('milliseconds_per_item','mean')).reset_index()
        eligible=aggregate[aggregate.accuracy>=aggregate.accuracy.max()-.01]
        chosen=eligible.sort_values(['latency','model']).iloc[0]['model']
        # Deployment checkpoint is predeclared seed 42, rather than selecting a lucky seed.
        run=df[(df.model==chosen)&(df.seed==42)].iloc[0]
        manifest[task]={'model':run.model_path,'revision':None,'base_model':chosen,'base_revision':run.revision,'seed':42,'split_sha256':run.split_sha256}
        summary.extend([{'task':task,**x,'selected':x['model']==chosen} for x in aggregate.to_dict('records')])
    Path(args.out).write_text(json.dumps(manifest,indent=2))
    Path(args.out).with_name('selection_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
