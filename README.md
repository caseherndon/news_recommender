# Problem Formulation

## 1. User-item interaction matrix

In this model, we will have $m$ users and $n$ news items. The observed user-item interactions will be represented by an $m\times n$ matrix where an element is 1 if user $u$ clicked on news item $i$ and 0 otherwise. However, a 0 does not necessarily mean that a user would not click on a particular news item given the opportunity; they might just not have been exposed to it. Therefore, we will consider two additional matrices: an exposure matrix $E$ and a latent preference matrix $Y$, both $m\times n$ in size. The exposure matrix is 1 if user $u$ was exposed to news item $i$ regardless of whether they clicked on it and 0 otherwise. The latent preference matrix is 1 if the user $u$ would click on news item $i$ when exposed to it and 0 otherwise. Thus, the formulation of these matrices is $R = E\odot Y$, where $\odot$ is the Hadamard product, meaning each element of $R$ is the product of the corresponding entries in $E$ and $Y$. We can observe $R$ and $E$, but $Y$ will need to be estimated through the recommender system.

## 2. Low-Rank Latent Factor Model

We will consider $S$ as an $m\times n$ latent preference score matrix, where each element of $S$ is the predicted preference score of user $u$ for news item $i$. Because we assume that user preferences and news item features can be represented by a small number of latent factors, we can say that $S$ has the structure $S=PQ^{\small{\textrm{T}}}$. $P$ is the $m\times k$ matrix for the user preferences, and $Q$ is the $n\times k$ matrix for the news item features. Since $S$ is the product of two matrices with $k$ columns/rows, $\textrm{rank}(S)\leq k$. Each element of $S$ can be found by taking the dot product of the $u$-th row of $P$ and $i$-th row of $Q$, representing $k$ "features" of user $u$ and news item $i$ respectively. We impose this low-rank structure because it lets the large sparse user-item matrix be approximated through lower-dimensional user and item embeddings, rather than treating $S$ as an arbitrary matrix with no shared structure. This is the mathematical basis of collaborative filtering: the model learns user and item representations from collective interaction patterns rather than relying on explicit user features or item content.

## 3. Implicit Matrix Factorization Objective

The goal is to find user and news item latent factor matrices $P$ and $Q$ by minimizing the following loss function
$$\sum_{u,i} c_{ui} (R_{ui}-P^{(u)}\cdot Q^{(i)})^{2}+\lambda (\|P\|_{F}^{2}+\|Q\|_{F}^{2}).
$$

We will consider each term of this loss function in turn.

$(R_{ui}-P^{(u)}\cdot Q^{(i)})^{2}$ measures the prediction error for each user $u$ and news item $i$. $R_{ui}$ is the observed value and $P^{(u)}\cdot Q^{(i)}$ is the predicted value from the low-rank factorization, so the overall value will be small when the predicted value is close to the observed value.

$c_{ui}$ is the confidence weight, defined by $c_{ui}=1+\alpha R_{ui}$, where $\alpha$ is greater than zero. If the element is unclicked, the confidence weight is 1, so the unclicked element is treated as a low-confidence observation. If the element is clicked, the confidence weight becomes larger, so the clicked element receives a larger weight. Because a click is known, observed data, this reflects the idea that we want to strongly reward the model for correctly recovering clicked interactions. An unclicked element, however, is treated more cautiously. Because we are trying to recover the latent preference matrix rather than simply fit the observed clicks, we do not want to penalize the model too heavily for unclicked news items; that user may have never been exposed to that news item in the first place. The model therefore pays more attention to clicked interactions but still uses unclicked elements as weak information.

$\lambda (\|P\|_{F}^{2}+\|Q\|_{F}^{2})$ is a regularization term. The squared Frobenius norms $\|P\|_{F}^{2}$ and $\|Q\|_{F}^{2}$ measure the total squared magnitude of all elements in the user and item latent factor matrices. This term penalizes overly large latent factors and helps prevent overfitting. Without it, the model might fit the training data too closely by learning very large or highly specialized embeddings. $\lambda$ controls the strength of this penalty.



## 4. Explicit feedback and implicit feedback

In implicit feedback recommendation, we sum over all elements of the observed user-item interaction matrix rather than only the non-zero elements. This differs from the standard explicit-feedback matrix factorization objective, such as movie ratings, where the loss is usually computed only over observed ratings. Missing ratings are truly unknown and should not be treated as zero. In implicit feedback, however, a clicked element is a strong positive signal, but an unclicked element is not necessarily a strong negative signal. It may mean the user disliked the item, but it may also mean the user was never exposed to it. Therefore, we should not simply discard all zero elements. If we only trained on clicked elements, the model would learn to predict high values for those items, but it would have no signal telling it to predict lower values for unclicked items. By considering all user-item pairs and weighting them by confidence, the model is encouraged to assign high scores to clicked items while keeping the scores of unclicked items relatively low, but only with low confidence.