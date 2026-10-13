import argparse,json,time,platform,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from datasets import Dataset
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,set_seed
from huggingface_hub import HfApi
from .common import CONFIG
def main():
    a=argparse.ArgumentParser();a.add_argument('--task',choices=['event','sentiment'],required=True);a.add_argument('--model',required=True);a.add_argument('--seed',type=int,default=42);a.add_argument('--data',default='data');a.add_argument('--out',required=True);a.add_argument('--reference',action='store_true');args=a.parse_args()
    set_seed(args.seed);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    labels=CONFIG['labels'] if args.task=='event' else ['negative','neutral','positive']
    frames={s:pd.read_csv(Path(args.data)/f'{args.task}_{s}.csv') for s in ['train','validation']}
    if set(frames['train'].id)&set(frames['validation'].id):raise ValueError('Split leakage')
    lock=json.loads((Path(__file__).resolve().parents[1]/'source_lock.json').read_text())
    if args.model not in lock:raise ValueError('Model is not in the verified candidate lock')
    rev=lock[args.model]['revision']
    tok=AutoTokenizer.from_pretrained(args.model,revision=rev)
    kwargs={} if args.reference else dict(num_labels=len(labels),id2label=dict(enumerate(labels)),label2id={l:i for i,l in enumerate(labels)})
    model=AutoModelForSequenceClassification.from_pretrained(args.model,revision=rev,**kwargs)
    if args.reference and args.task!='sentiment':raise ValueError('Reference only supported for sentiment')
    def ds(frame):
        t=Dataset.from_dict({'text':frame.text.tolist(),'labels':[labels.index(x) for x in frame.label]})
        return t.map(lambda b:tok(b['text'],truncation=True,max_length=CONFIG['max_length']),batched=True).remove_columns('text')
    def metrics(p):return {'accuracy':accuracy_score(p.label_ids,np.argmax(p.predictions,axis=-1)),'macro_f1':f1_score(p.label_ids,np.argmax(p.predictions,axis=-1),average='macro')}
    trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=CONFIG['epochs'],per_device_train_batch_size=CONFIG['batch_size'],per_device_eval_batch_size=CONFIG['batch_size'],learning_rate=CONFIG['learning_rate'],eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='accuracy',report_to='none',seed=args.seed,save_total_limit=1),train_dataset=ds(frames['train']),eval_dataset=ds(frames['validation']),processing_class=tok,data_collator=DataCollatorWithPadding(tok),compute_metrics=metrics)
    start=time.perf_counter()
    if not args.reference:trainer.train()
    train_seconds=time.perf_counter()-start
    trainer.save_model(str(out/'model'));tok.save_pretrained(out/'model')
    # Shared inference path: reference model label order is read from its config.
    from .inference import classify
    classify(str(out/'model'),frames['validation'].text.iloc[:8].tolist())
    times=[]
    for _ in range(3):
        start=time.perf_counter();pred=classify(str(out/'model'),frames['validation'].text.tolist());times.append(time.perf_counter()-start)
    result={'task':args.task,'model':args.model,'revision':rev,'seed':args.seed,'stage':'validation','accuracy':accuracy_score(frames['validation'].label,[x['label'] for x in pred]),'macro_f1':f1_score(frames['validation'].label,[x['label'] for x in pred],average='macro'),'train_seconds':train_seconds,'inference_seconds_median':float(np.median(times)),'milliseconds_per_item':1000*float(np.median(times))/len(pred),'n':len(pred),'device':str(trainer.args.device),'platform':platform.platform(),'torch':torch.__version__,'split_sha256':{s:hashlib.sha256((Path(args.data)/f'{args.task}_{s}.csv').read_bytes()).hexdigest() for s in frames}}
    (out/'results.json').write_text(json.dumps(result,indent=2))
    result['training_device']=result.pop('device');result['inference_device']='cpu'
    (out/'results.json').write_text(json.dumps(result,indent=2))
    predicted=[x['label'] for x in pred]
    pd.DataFrame({'text':frames['validation'].text,'truth':frames['validation'].label,'prediction':predicted}).to_csv(out/'validation_predictions.csv',index=False)
    pd.DataFrame(confusion_matrix(frames['validation'].label,predicted,labels=labels),index=labels,columns=labels).to_csv(out/'validation_confusion_matrix.csv')
if __name__=='__main__':main()
