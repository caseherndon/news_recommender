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