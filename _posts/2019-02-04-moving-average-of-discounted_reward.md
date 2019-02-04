---
layer: post
title: "Moving Average of Discounted Reward"
author: "Wei Xu"
categories: [machine-learning]
---

The sum of discounted rewards at step t is:

$$ R_t = \sum_{s=t}^\infty \gamma^{s-t} r_s $$

We want to calculate the exponential moving average of \\(R_t\\)

\begin{equation}
\bar{R}_t = \alpha \bar{R}_{t-1} + (1-\alpha) R_t \label{EqUnbiasedR}
\begin{equation}

Let \\(\alpha_s^t=\prod_{k=s+1}^t \alpha_k\\). To make an unbiased estimation of \\(\bar{R} = E(R)\\), we need to correct it by:

$$ \bar{R} = \frac{\bar{R}_t}{1-\alpha_0^t} $$

Now we develop a tractable procedure to calculate \\(\bar{R}_t\\):

$$
\begin{align}
\bar{R}_t &= \alpha \bar{R}_{t-1} + (1-\alpha) R_t = \sum_{s=1}^t (1-\alpha_s)\alpha_s^t R_s \\
&= \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \sum_{k=s}^\infty \gamma^{k-s}r_k = \sum_{s=1}^t \sum_{k=s}^\infty (1-\alpha_s)\alpha_s^t \gamma^{k-s}r_k \\
&= \sum_{k=1}^\infty r_k \sum_{s=1}^{\min(k,t)} (1-\alpha_s)\alpha_s^t \gamma^{k-s} \\
&= \sum_{k=1}^t r_k \sum_{s=1}^k(1-\alpha_s) \alpha_s^t \gamma^{k-s} + \sum_{k=t+1}^\infty r_k \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \gamma^{k-s}\\
&= \sum_{k=1}^t r_k \alpha_k^t\sum_{s=1}^k (1-\alpha_s)\alpha_s^k \gamma^{k-s} + \sum_{k=t+1}^\infty r_k\gamma^{k-t} \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \gamma^{t-s}\\
&= \sum_{k=1}^t r_k \alpha_k^t c_k + \sum_{k=t+1}^\infty r_k\gamma^{k-t} c_t = \sum_{k=1}^t r_k \alpha_k^t c_k + \gamma c_t \sum_{k=t+1}^\infty r_k\gamma^{k-t-1} \\
&= \sum_{k=1}^t r_k \alpha_k^t c_k + \gamma c_t R_{t+1} \label{EqBarRt} \\
\end{align}
$$

where \\( c_t = \sum_{s=1}^t (1-\alpha_s)\alpha_s^t \gamma^{t-s} \\)

Let \\(\hat{R}_t = \sum_{k=1}^t r_k \alpha_k^t c_k\\). We can calculate \\(\hat{R}_t\\) using the following procedure:

$$
\begin{align}
	& c_0 = 0\\
	& \hat{R}_0 = 0 \\
	& c_t = \alpha_t\gamma_t c_{t-1} + 1-\alpha_t \\
	& \hat{R}_t = \alpha \hat{R}_{t-1} + c_t r_t \\
	& \mbox{Reset } c_t \mbox{ to 0 if at the end of an episode} \\
\end{align}  
$$

By (\ref{EqBarRt}), the difference between $\bar{R}_t$ and $\hat{R}_t$ is:

$$ \bar{R}_t - \hat{R}_t = \gamma c_t R_{t+1} $$

If the sequence of $\alpha_t$ have the following property:
$$
	(1-\alpha_s)\alpha_s^t \le 1-\alpha_t
$$
we can bound the \\(c_t\\) by

$$
c_t \le \sum_{s=1}^t (1-\alpha_t) \gamma^{t-s} < \frac{1-\alpha_t}{1-\gamma}
$$

This requires \\(\alpha_t\\) satisfy the following condition:

\begin{equation*}
	& (1-\alpha_s)\alpha_s^t \le 1-\alpha_t \\
	& \frac{1-\alpha_s}{\alpha_0^s} \le \frac{1-\alpha_t}{\alpha_0^t} \\
	& \frac{1-\alpha_{t-1}}{\alpha_0^{t-1}} \le \frac{1-\alpha_t}{\alpha_0^t} \\
	& \alpha_t \le \frac{1}{2-\alpha_{t-1}} \\
\end{equation*}  


The difference between \\(\bar{R}_t\\) and \\(\hat{R}_t\\) is:

$$ \bar{R}_t - \hat{R}_t = \gamma c_t R_{t+1} < \frac{1-\alpha_t}{1-\gamma} \gamma R_{t+1} $$

If \\(1-\alpha_t\\) is much smaller than \\(1-\gamma\)), this difference is negligible.
Or we can correct it by assuming \\(R_{t+1}=\bar{R}\\). Using (\eqref{EqUnbiasedR}), we get:

$$ (1- \alpha_0^t)\bar{R}  - \hat{R}_t = \gamma c_t \bar{R} $$

This gives us a biased corrected estimator of \\(E(R)\\):

$$ \bar{R} = \frac{\hat{R}_t}{1-\alpha_0^t - \gamma c_t} $$
