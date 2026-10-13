---
language: en
library_name: transformers
pipeline_tag: text-classification
---
# FinSignal Group 7 selected model (complete before publishing)
Base checkpoint and immutable revision: pending selection.
Training source and source revision: source_lock.json plus reviewed event annotation where applicable.
Labels: document exact config.id2label values.
Training: epochs 3, learning rate 2e-5, max length 256, batch size 16, seed 42. Report actual environment and GPU/CPU.
Validation accuracy/runtime and three-seed variation: pending measured results.
Held-out accuracy: pending final app evaluation.
License: specify an appropriate license based on base weights and source terms. Do not use an unverified default. FinBERT reference retains its source license and attribution.
Intended use: academic financial-news research prototype for ISOM5240, focused on the Wealthsimple business case. No affiliation or commercial endorsement claimed.
Limitations: short-text source, seven-category single-event simplification, 256-token truncation, uncalibrated confidence, potential domain shift and label ambiguity. Not investment advice.
