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