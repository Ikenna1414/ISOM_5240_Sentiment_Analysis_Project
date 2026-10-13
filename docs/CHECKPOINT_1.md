# Checkpoint 1 - sources and candidate models
The official proposal has been read and retained. The submission reminder image was inspected. The full course requirements document was not among retrieved attachments.

Sources are pinned in source_manifest.json/source_lock.json. Downloaded counts: 21,107 topic items and 11,931 sentiment items. Card counts differ slightly from actual CSVs; the snapshot manifest is authoritative for experiments.
After cleaning, conflicting-label removal and deduplication, sentiment splits are train 8,015 / validation 1,415 / test 2,345. These pass exact-text separation checks. Near-duplicate/story review remains required for stronger independence.
The event review queue contains 1,922 items sampled across original topic classes. No reviewed event labels or fine-tuned event model exist yet. Two-annotator review and adjudication are the next substantive step.

Locked candidates: DistilBERT and RoBERTa for both tasks, plus unchanged FinBERT sentiment reference. Event labels follow the proposal's seven categories. Highest mean validation accuracy wins; within 0.01 absolute accuracy choose lower CPU latency. Deploy the selected candidate's predeclared seed-42 checkpoint.

Implementation prepared: data acquisition, annotation protocol, group splits, candidate training, shared CPU inference, single/batch Streamlit UI, selection, publishing, held-out evaluation, result export, eight unexecuted notebooks and submission folders.
Validation performed: Python/notebook syntax, prepared sentiment class coverage and exact overlap, event group separation, rejection of unreviewed labels. Transformer training and real-model app behavior have not yet been exercised.
The Excel file is a pending ledger with blank scores; it is not experimental evidence. The report and presentation plan are drafts. Training, actual model scores, final report/slides/video and public URLs remain outstanding.
