import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .common import CONFIG,key
def main():
    a=argparse.ArgumentParser();a.add_argument('annotations');a.add_argument('--out',default='data');args=a.parse_args()
    df=pd.read_csv(args.annotations).fillna('')
    if not df.review_status.eq('approved').all():raise ValueError('Every included row must be reviewed and approved; filter excluded rows first.')
    if not df.event_label.isin(CONFIG['labels']).all():raise ValueError('Invalid event label')
    if df.story_group.eq('').any():raise ValueError('Supply a story group to keep related headlines together')
    df['id']=df.text.map(key)
    if df.groupby('id').event_label.nunique().max()>1:raise ValueError('Conflicting duplicate labels')
    df=df.drop_duplicates('id');df['label']=df.event_label
    train_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=df.story_group))
    pool=df.iloc[train_idx];test=df.iloc[test_idx]
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.1875,random_state=43).split(pool,groups=pool.story_group))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for split,t in [('train',pool.iloc[tr]),('validation',pool.iloc[va]),('test',test)]:
        if set(t.label)!=set(CONFIG['labels']):raise ValueError(f'{split} lacks classes; acquire more examples before training')
        t.to_csv(out/f'event_{split}.csv',index=False)
    (out/'event_split_counts.json').write_text(json.dumps({s:pd.read_csv(out/f'event_{s}.csv').label.value_counts().to_dict() for s in ['train','validation','test']},indent=2))
if __name__=='__main__':main()
