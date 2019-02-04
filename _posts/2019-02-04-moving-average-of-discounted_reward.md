---
layer: post
title: "Moving Average of Discounted Reward"
author: "Wei Xu"
categories: [machine-learning]
---

The sum of discounted rewards at step t is:
$$ R_t = \sum_{s=t}^\infty \gamma^{s-t} r_s $$

We want to calculate the exponential moving average of $$R_t$$
$$
\bar{R}_t = \alpha \bar{R}_{t-1} + (1-\alpha) R_t \label{EqUnbiasedR}
$$
Let $\alpha_s^t=\prod_{k=s+1}^t \alpha_k$. To make an unbiased estimation of $$\bar{R} = E(R)$$, we need to correct it by:
$$ \bar{R} = \frac{\bar{R}_t}{1-\alpha_0^t} $$
Now we develop a tractable procedure to calculate $$\bar{R}_t$$:
$$
\begin{align}
\bar{R}_t &=& \alpha \bar{R}_{t-1} + (1-\alpha) R_t = \sum_{s=1}^t (1-\alpha_s)\alpha_s^t R_s \\
&=& \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \sum_{k=s}^\infty \gamma^{k-s}r_k = \sum_{s=1}^t \sum_{k=s}^\infty (1-\alpha_s)\alpha_s^t \gamma^{k-s}r_k \\
&=& \sum_{k=1}^\infty r_k \sum_{s=1}^{\min(k,t)} (1-\alpha_s)\alpha_s^t \gamma^{k-s} \\
&=& \sum_{k=1}^t r_k \sum_{s=1}^k(1-\alpha_s) \alpha_s^t \gamma^{k-s} + \sum_{k=t+1}^\infty r_k \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \gamma^{k-s}\\
&=& \sum_{k=1}^t r_k \alpha_k^t\sum_{s=1}^k (1-\alpha_s)\alpha_s^k \gamma^{k-s} + \sum_{k=t+1}^\infty r_k\gamma^{k-t} \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \gamma^{t-s}\\
&=& \sum_{k=1}^t r_k \alpha_k^t c_k + \sum_{k=t+1}^\infty r_k\gamma^{k-t} c_t = \sum_{k=1}^t r_k \alpha_k^t c_k + \gamma c_t \sum_{k=t+1}^\infty r_k\gamma^{k-t-1} \\
&=& \sum_{k=1}^t r_k \alpha_k^t c_k + \gamma c_t R_{t+1} \label{EqBarRt} \\
\end{align}
$$
where
$$ c_t = \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \gamma^{t-s} $$
