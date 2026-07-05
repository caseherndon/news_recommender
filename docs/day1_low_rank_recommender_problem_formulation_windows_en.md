# Day 1: Problem Formulation for Low-Rank Recommendation Systems and a Minimal Runnable SVD Example

## Today's Positioning

Establish the project skeleton, clearly write the mathematical problem formulation, and run through a minimal low-rank recommendation toy example.

The long-term main line of this project is:

> sparse and biased user-item interaction matrix  
> → low-rank matrix factorization  
> → top-K ranking evaluation  
> → exposure bias / popularity bias analysis  
> → two-stage retrieval + ranking  
> → demo / API / optional Azure deployment

Day 1 only does the first part:

> user-item matrix → low-rank structure → SVD toy recommender → README mathematical narrative

---

## Final Deliverables for Today

By the end of today, the project repo should have at least the following files:

```text
recsys-news-debiasing/
├── README.md
├── requirements.txt
├── docs/
│   └── 01_problem_formulation.md
├── notebooks/
│   └── 01_low_rank_recommender_toy_example.ipynb
└── src/
    ├── data/
    ├── models/
    ├── evaluation/
    ├── retrieval/
    └── ranking/
```

The minimum completion standards for today:

1. The project directory has been created.
2. `requirements.txt` has been written and can be installed.
3. `docs/01_problem_formulation.md` has a first version of the mathematical formulation.
4. `notebooks/01_low_rank_recommender_toy_example.ipynb` runs through the toy SVD recommendation.
5. `README.md` has clear project positioning and current progress.

---

## Task 1: Create the Project Structure

Run locally:

```powershell
Set-Location "C:\Users\hl\Desktop\PythonProject1\mind"

mkdir docs, notebooks, src\data, src\models, src\evaluation, src\retrieval, src\ranking

type nul > README.md
type nul > requirements.txt
type nul > docs\01_problem_formulation.md
```

It is recommended not to create too many complex files today. The directories can exist first, and code files can be added gradually later.

---

## Task 2: Write requirements.txt

Today, only install the dependencies needed for the mathematical baseline:

```txt
numpy
pandas
scipy
scikit-learn
matplotlib
jupyter
ipykernel
```

Set up the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name recsys-news-debiasing
```

No need to install today:

- `implicit`
- `lightfm`
- `faiss`
- `torch`
- `sentence-transformers`
- `fastapi`
- `streamlit`
- Azure-related SDKs

All of these are left for later stages.

---

## Task 3: Write docs/01_problem_formulation.md

This document is the mathematical foundation of the project. Today it does not need to be written like a paper, but the core concepts must be explained clearly.

The suggested structure is as follows:

```md
# Problem Formulation

## 1. Recommendation as Matrix Completion

Let there be \(m\) users and \(n\) news items. Define the user-item interaction matrix

\[
R \in \{0,1\}^{m \times n}.
\]

Here, \(R_{ui}=1\) means user \(u\) clicked item \(i\). However, in implicit feedback recommendation, \(R_{ui}=0\) does not necessarily mean the user dislikes the item. It may simply mean the user was never exposed to it.

## 2. Exposure Bias

A more accurate observation model is

\[
R_{ui} = E_{ui}Y_{ui},
\]

where:

- \(E_{ui}=1\) means item \(i\) was exposed to user \(u\);
- \(Y_{ui}=1\) means user \(u\) would click or like item \(i\);
- \(R_{ui}\) is the observed click.

In matrix form:

\[
R = E \odot Y.
\]

The recommender system wants to learn the hidden preference matrix \(Y\), but only observes the biased click matrix \(R\).

## 3. Low-Rank Latent Factor Model

Assume each user and item can be represented in a shared \(k\)-dimensional latent space:

\[
p_u \in \mathbb{R}^k, \quad q_i \in \mathbb{R}^k.
\]

The predicted preference score is

\[
s(u,i) = p_u^\top q_i.
\]

In matrix form:

\[
S = P Q^\top,
\]

where \(P\) is the user embedding matrix and \(Q\) is the item embedding matrix. Since \(S = P Q^\top\), we have

\[
\operatorname{rank}(S) \leq k.
\]

This is the low-rank structure behind collaborative filtering.

## 4. Implicit Matrix Factorization Objective

For implicit feedback, one common objective is

\[
\min_{P,Q}
\sum_{u,i} c_{ui}(R_{ui} - p_u^\top q_i)^2
+
\lambda(\|P\|_F^2 + \|Q\|_F^2),
\]

where \(c_{ui}\) is a confidence weight. A common choice is

\[
c_{ui} = 1 + \alpha R_{ui}.
\]

Clicked interactions receive higher confidence, while unclicked interactions are treated as low-confidence observations rather than strong negatives.
```

Today's focus:

- Be sure to write \(R = E \odot Y\).
- Be sure to write \(S = P Q^\top\).
- Be sure to explain why unclicked does not equal dislike.
- Be sure to explain why this is a low-rank problem.

---

## Task 4: Create the Toy SVD Notebook

Create:

```text
notebooks/01_low_rank_recommender_toy_example.ipynb
```

The goal of the notebook is not to pursue performance, but to demonstrate the mathematical intuition of recommender systems.

### 4.1 Construct a Toy User-Item Matrix

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

users = ["u1", "u2", "u3", "u4", "u5"]
items = ["Politics", "Tech", "Sports", "Finance", "Health", "Travel"]

R = np.array([
    [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 0],
])

df = pd.DataFrame(R, index=users, columns=items)
df
```

### 4.2 Do Truncated SVD

```python
k = 2

svd = TruncatedSVD(n_components=k, random_state=42)
P = svd.fit_transform(R)
Q = svd.components_.T

S_hat = P @ Q.T
score_df = pd.DataFrame(S_hat, index=users, columns=items)
score_df
```

### 4.3 Generate Top-K Recommendations

```python
def recommend_for_user(user_id, R, scores, users, items, top_k=3):
    u_idx = users.index(user_id)
    seen = R[u_idx] > 0

    user_scores = scores[u_idx].copy()
    user_scores[seen] = -np.inf

    top_indices = np.argsort(user_scores)[::-1][:top_k]
    return [(items[i], float(user_scores[i])) for i in top_indices]

recommend_for_user("u1", R, S_hat, users, items, top_k=3)
```

### 4.4 Explanation to Write in the Notebook

Write in a notebook markdown cell:

```md
The original user-item matrix is sparse. However, users and news items may share a low-dimensional latent structure. For example, users who click Politics and Tech may be close in latent space, while users who click Sports and Health may form another group.

Truncated SVD approximates the sparse interaction matrix with a low-rank score matrix. The reconstructed scores can be used to recommend unseen items with high predicted preference.
```

Chinese interpretation:

> The original click matrix is very sparse, but there may be a low-dimensional interest structure between users and news. SVD recovers this latent structure through low-rank approximation, so that it can recommend news categories that the user has not clicked but may be interested in.

---

## Task 5: Write the First Version of README.md

The README only needs to state the project positioning today. It does not need to include complete experimental results.

Suggested content:

```md
# Debiased News Recommendation via Low-Rank Matrix Factorization and Counterfactual Evaluation

This project builds a mathematically grounded news recommendation system based on low-rank matrix factorization, ranking-based evaluation, and counterfactual-style debiasing.

The project treats recommendation not simply as click prediction, but as learning latent user-item preference structure from sparse and biased implicit feedback data.

## Core Ideas

- User-item interaction matrix
- Low-rank matrix factorization
- Implicit feedback modeling
- Exposure bias and popularity bias
- Ranking metrics such as Recall@K, NDCG@K, MRR, and MAP
- Counterfactual-style offline evaluation using IPS and SNIPS
- Two-stage retrieval and ranking architecture

## Mathematical View

Given a user-item interaction matrix

\[
R \in \{0,1\}^{m \times n},
\]

we model user preferences using a low-rank score matrix

\[
S = P Q^\top.
\]

Observed clicks are biased because users only interact with exposed items:

\[
R = E \odot Y.
\]

## Current Progress

- [x] Project formulation
- [x] Toy low-rank recommender
- [ ] MIND-small data processing
- [ ] Popularity baseline
- [ ] ItemKNN / UserKNN
- [ ] Truncated SVD baseline
- [ ] ALS
- [ ] BPR
- [ ] LightFM
- [ ] Ranking evaluation
- [ ] Counterfactual evaluation
- [ ] Two-stage retrieval and ranking
- [ ] FastAPI / Streamlit demo
```

---

## Things Not to Do Today

To avoid the project getting out of control at the beginning, explicitly do not do the following today:

1. Do not download MIND-full.
2. Do not do Azure.
3. Do not write FastAPI.
4. Do not write Streamlit.
5. Do not do two-tower.
6. Do not do LLM embedding.
7. Do not do complex debiased evaluation.
8. Do not get stuck on model metrics.
9. Do not introduce too many dependencies.
10. Do not write the README as a vague project introduction.

Today, only do the mathematical foundation and the minimal runnable demo.

---

## Self-Check Questions After Completing Today

After finishing Day 1, you should be able to answer these questions:

1. Why can a recommender system be viewed as a user-item matrix problem?
2. Why does \(R_{ui}=0\) not necessarily mean that the user does not like the item?
3. What do \(E\) and \(Y\) mean respectively in \(R = E \odot Y\)?
4. Why does matrix factorization correspond to low-rank structure?
5. What are \(P\) and \(Q\) respectively in \(S = P Q^\top\)?
6. How does the SVD toy example generate recommendations?
7. Why not directly do deep learning today?

If all of these questions can be explained clearly, then Day 1 is successful.

---

## Suggested Git Commit at the End of Today

If you use Git, at the end of today you can commit:

```powershell
git init
git add .
git commit -m "Day 1: initialize low-rank recommender foundation"
```

---

## Day 1 Success Standard

Today is not about proving model effectiveness, but about proving that the project direction is valid.

The sign of Day 1 success is:

> This project already has a clear mathematical problem formulation, a runnable low-rank recommendation toy example, and a repo skeleton that can continue to expand to real MIND data.

The next step, Day 2, is to enter:

> MIND-small data download, parsing `behaviors.tsv` / `news.tsv`, and user-item interaction matrix construction.
