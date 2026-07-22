# Pitfalls Log

## 7/20/26 - Merging only history rows into train_with_news silently drops users

An earlier version of the interaction-parsing notebook built `train_with_news` by merging `interactions_history` alone with news metadata, omitting `interactions_impression` entirely. This silently dropped any user whose only train signal was a clicked impression (no browsing history) — 892 such users in MIND-small train. Lesson: when a table is split into multiple parsed sources (history vs. impression), double check the final merge point actually concatenates all intended sources — a partial merge won't error, it will just quietly shrink the dataset.

## 7/20/26 - Unfiltered click=0 rows can leak into "positive" interaction tables

Because `interactions_history`/`interactions_impression` are combined into one interaction table before any click filtering, `train_with_news` and `dev_with_news` contain both click=0 and click=1 impression rows. If sparse matrix construction (Day 5, Step 9) reads directly from these tables without an explicit `click == 1` filter, non-clicked exposures get treated as positive interactions. Lesson: verify the exact filter condition used at each stage that assumes "positive interaction only" — don't assume it was already applied upstream just because the table is named `..._with_news`.

## 7/21/26 - Identical recommendations across sample users can look like a bug but isn't

When several sample users in Cell 8 all received the same top-K popularity recommendations, it initially looked like a cold-start issue (all users defaulting to an unpersonalized list). In fact, `sample_users` is drawn from `user_idx_map.keys()`, which is built from the train matrix, so all sampled users are guaranteed to be train (warm-start) users, not cold-start. The identical output is instead a natural consequence of the popularity baseline: exclusion only changes a user's list if they've personally clicked one of the top-K popular items, so users with no overlap in their seen items will get exactly the same list. Worth remembering when sanity-checking future models: identical outputs don't always mean a fallback path was hit.