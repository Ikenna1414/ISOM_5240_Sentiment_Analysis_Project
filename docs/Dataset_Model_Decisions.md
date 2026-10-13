# Checkpoint 1 - verified design, 13 October 2026
Source of truth: official two-page proposal, 5 October 2026. Event dataset fallback is explicitly permitted.
Locked acquisition sources: zeroshot/twitter-financial-news-sentiment and zeroshot/twitter-financial-news-topic.
Locked trainable candidates for both tasks: distilbert/distilbert-base-uncased and FacebookAI/roberta-base. Unchanged sentiment reference: ProsusAI/finbert.
Final selected models remain undecided until measured validation accuracy/runtime exists. Event source needs reviewed project labels; broad topic mapping is not accepted as ground truth.
Evaluation: validation accuracy plus measured CPU runtime for selection; frozen held-out accuracy with deployed model revisions for application assessment. Report macro F1, class counts, confusion matrices and failure examples as diagnostics.
Sources checked:
https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic
https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
https://huggingface.co/ProsusAI/finbert
https://huggingface.co/distilbert/distilbert-base-uncased
https://huggingface.co/FacebookAI/roberta-base
https://huggingface.co/docs/transformers/training
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
