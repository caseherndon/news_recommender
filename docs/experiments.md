# Experiments Log

## 7/17/26 - Toy Model Alteration

When setting up the toy low-rank recommender, I decided to alter the R matrix to observe the changes to the final recommendations, as well as the P and Q matrices. This allowed me to understand better the intuition behind the k-rank factorization.

## 2026-07-17 — Initial MIND-small raw data inspection

**Dataset version / split:**
- MINDsmall_train (behaviors.tsv, news.tsv)
- MINDsmall_dev (behaviors.tsv)

**Train behaviors.tsv:**
- Raw rows: 156,965
- Unique users: 50,000
- Clicked impression entries: 236,344
- Non-clicked (exposed) impression entries: 5,607,100

**Train news.tsv:**
- Unique news items: 51,282

**Dev behaviors.tsv:**
- Unique users: 50,000
- Clicked impression entries: 111,383
- Non-clicked (exposed) impression entries: 2,629,615
- (Raw row count not captured in this notebook — only nunique/click counts were printed)

**Cold-start overlap:**
- Dev users only in dev (not in train): 44,057 out of 50,000 dev users

**Interpretation:**
Only 5,943 of the 50,000 dev users (about 12%) also appear in train — a much smaller warm-start overlap than initially expected. This confirms that strict warm-start filtering is essential before evaluation, and that cold-start coverage (Popularity fallback / UserKNN borrowing) will affect the large majority of dev users rather than a small minority.

## 2026-07-20 — Day 5 user count discrepancy investigation

**Observation:**
`train_with_news["user_id"].nunique()` returns 49,108, vs. 50,000 unique users seen in the raw `behaviors.tsv` inspection on 7/17.

**Investigation:**
- Identified 892 users present in raw train behaviors but absent from `train_with_news`.
- Sample check (`user 17674`) showed `history = NaN` and impressions including a clicked entry (`N56214-1`).
- Traced root cause to an earlier version of the parsing notebook (`03_parse_impression_small.ipynb`) where `train_with_news` was merged from `interactions_history` only, silently dropping all clicked-impression-only users (no history, but a click=1 impression).
- Confirmed the current production script (`03_parse_impression_history.py` → `04_merge_news_metadata.py`) does *not* have this issue — both impression rows (click 0 and 1) and history rows are combined into one table before the news merge.
- Found a separate, still-open issue: `train_with_news`/`dev_with_news` in the current pipeline retain click=0 impression rows unfiltered, which would inflate the sparse matrices with non-positive interactions if not filtered before Step 9 of Day 5.

**Interpretation:**
The 49,108 vs 50,000 discrepancy stems from a bug in an earlier, non-production version of the parsing step, not the current pipeline. The current pipeline instead needs an explicit click-filter step before matrix construction (see decisions.md 7/20/26).