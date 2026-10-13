# GitHub setup

Suggested repository name: finsignal-group07.

1. Create a GitHub repository under your account. The proposal calls for a public project repository. Do not initialize with extra files when importing this prepared folder.
2. Upload the CONTENTS of FinSignal-GitHub so app.py, README.md and requirements.txt are directly at the repository root. Retain the finsignal, notebooks, docs, examples, results and .streamlit folders. Enable hidden files in your file manager to include .gitignore and .streamlit/config.toml.
3. Confirm the repository home page displays the README and root app.py. The initial commit can be titled "Add FinSignal Group 7 prototype and reproducible experiments".
4. Run the data review and notebook experiments before enabling model predictions. Keep large datasets, weights, tokens and local environments out of GitHub. Publish selected weights/tokenizers/model cards on Hugging Face.
5. Add measured experiment CSVs, model-selection rationale and final documentation when available.
6. After Hugging Face publication, configure deployment_manifest.json with published model IDs and exact commit revisions. Remove its entry from .gitignore and commit it. Do not commit a manifest containing local computer paths or credentials.
7. In Streamlit Community Cloud, connect GitHub, select this repository and branch, set app.py as the entry point, and use Python 3.11. Without a selected model manifest the app will show the pending state.
8. Record the actual Cloud URL and verify live single/batch outputs against frozen test predictions. Add the URL and measured deployed accuracy to the report and README only after verification.

The coursework submission archive is separate. Do not upload the entire Canvas/OneDrive package as the repository root. Final report, slides, video and model files still belong in their required submission folders.
