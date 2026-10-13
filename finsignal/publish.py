"""Explicitly run after selecting models; requires HF_TOKEN in the environment."""
import argparse,json
from pathlib import Path
from huggingface_hub import HfApi
def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',default='deployment_manifest.json');a.add_argument('--namespace',required=True);args=a.parse_args()
    path=Path(args.manifest);manifest=json.loads(path.read_text());api=HfApi()
    for task,entry in manifest.items():
        folder=Path(entry['model'])
        if not folder.is_dir():raise ValueError('Publish requires the locally selected model folder')
        if not (folder/'README.md').exists():raise ValueError('Add a model card with measured results, sources, labels and license before publication')
        repo=f'{args.namespace}/finsignal-group07-{task}';api.create_repo(repo,exist_ok=True)
        commit=api.upload_folder(repo_id=repo,folder_path=str(folder),commit_message='Publish selected Group 7 model and reproducibility evidence')
        entry['model']=repo;entry['revision']=commit.oid
    path.write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
