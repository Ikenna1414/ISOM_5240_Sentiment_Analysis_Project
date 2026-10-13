# Event annotation protocol
Label the primary explicitly reported company event. Read text only, never infer from stock movement or sentiment. Use these definitions:
- earnings_guidance: company financial results or forward financial outlook.
- mergers_acquisitions: acquisition, merger, disposal of a business, or takeover agreement/bid; ordinary portfolio investments are not automatically M&A.
- regulatory_legal: litigation, regulator investigation/action, legal ruling or compliance decision affecting a company.
- financing: equity/debt issuance, funding round, capital raising; treasury yields and dividends alone are other.
- management_change: appointment/departure of directors or senior management.
- product_business_expansion: launch, expansion, partnership or new market/facility; general company commentary is other.
- other: none of the six applies; macro news, analyst recommendations and stock-price moves alone belong here.

Two independent annotators record annotator_1/annotator_2. Resolve disagreements with a documented rationale into event_label. Set review_status=approved only after review; exclude ambiguous or truncated items and log why. Keep related reports of the same story under one story_group. Assign a unique group for unrelated items.
Measure raw agreement and Cohen's kappa before adjudication. Inspect class coverage and target at least 100 reviewed examples per event where available. Do not manufacture examples to fill missing classes. Expand the source queue or collect attributed public issuer headlines if coverage is insufficient. Record URLs, publication dates, source permissions and inclusion criteria for added sources.
The exported source topic is a sampling stratum, never an annotation shortcut. Queue balance does not imply event balance. Review near-duplicate wording before splitting. Freeze annotations and hashes before experiments.
