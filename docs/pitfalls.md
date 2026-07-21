# Pitfalls Log

## 7/20/26 - Merging only history rows into train_with_news silently drops users

An earlier version of the interaction-parsing notebook built `train_with_news` by merging `interactions_history` alone with news metadata, omitting `interactions_impression` entirely. This silently dropped any user whose only train signal was a clicked impression (no browsing history) — 892 such users in MIND-small train. Lesson: when a table is split into multiple parsed sources (history vs. impression), double check the final merge point actually concatenates all intended sources — a partial merge won't error, it will just quietly shrink the dataset.

## 7/20/26 - Unfiltered click=0 rows can leak into "positive" interaction tables

Because `interactions_history`/`interactions_impression` are combined into one interaction table before any click filtering, `train_with_news` and `dev_with_news` contain both click=0 and click=1 impression rows. If sparse matrix construction (Day 5, Step 9) reads directly from these tables without an explicit `click == 1` filter, non-clicked exposures get treated as positive interactions. Lesson: verify the exact filter condition used at each stage that assumes "positive interaction only" — don't assume it was already applied upstream just because the table is named `..._with_news`.