import argparse,json
from pathlib import Path
import pandas as pd
def main():
    a=argparse.ArgumentParser();a.add_argument('--runs',default='runs');a.add_argument('--out',default='Experimental_results.xlsx');args=a.parse_args()
    rows=[json.loads(p.read_text()) for p in Path(args.runs).glob('*/results.json')]
    if not rows:raise ValueError('No measured experiments; refusing to generate invented scores')
    df=pd.DataFrame(rows);df.to_excel(args.out,index=False);df.to_csv(Path(args.out).with_suffix('.csv'),index=False)
if __name__=='__main__':main()
