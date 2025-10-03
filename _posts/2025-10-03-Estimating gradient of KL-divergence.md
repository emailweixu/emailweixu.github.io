---
layer: post
title: "Estimating the gradient of KL-divergence"
author: "Wei Xu"
categories: [machine-learning]
---

For discrete distribution $$p$$ and $$q$$, the KL-divergence is:
$$KL(p \Vert q)=\sum_x p(x) \log (p(x)/q(x))=E_{x\sim p} \log(p(x)/q(x))$$.
So $$f_N(x)=\log(p(x)/q(x))$$ is an unbiased estimator of $$KL(p \Vert q)$$ if $$x\sim p$$.
Note that there are other estimators with lower variance, for example, $$f_S(x)=\log(p(x)/q(x))+(q(x)/p(x))-1$$ proposed
by [Schulman](http://joschu.net/blog/kl-approx.html).

If $$p$$ is parameterized with $$\theta$$ and $$f(x)$$ is an unbiased estimator
of $$KL(p\|q)$$, it is tempting to use $$\frac{\partial f(x)}{\partial \theta}$$ as an estimator for
$$\frac{\partial KL(p \Vert q)}{\partial \theta}$$. However, it is not an unbiased estimator
of $$\frac{\partial KL(p \Vert q)}{\partial \theta}$$ as previously noticed in [ALF](https://alf.readthedocs.io/en/latest/notes/estimating_derivative_of_expectation.html)

To get an unbiaded estimator of $$\frac{\partial KL(p \Vert q)}{\partial \theta}$$, we need to start from its definition:

$$
\begin{align}
\frac{\partial KL(p\|q)}{\partial \theta} &= \frac{\partial \sum_x p_\theta(x) \log (p_\theta(x)/q(x)) }{\partial \theta} \\
&= \sum_x \frac{\partial p_\theta(x)}{\partial \theta} \log (p_\theta(x)/q(x)) + p_\theta(x) \frac{\partial \log (p_\theta(x)/q(x)) }{\partial \theta} \\
&= \sum_x p_\theta(x) \frac{\partial \log(p_\theta(x)/q(x))}{\partial \theta} \log (p_\theta(x)/q(x)) + p_\theta(x) \frac{\partial \log (p_\theta(x)) }{\partial \theta} \\
&= E_{x\sim p} \frac{\partial \log(p_\theta(x)/q(x))}{\partial \theta} \log (p_\theta(x)/q(x)) + E_{x\sim p}\frac{\partial \log (p_\theta(x)) }{\partial \theta} \\
&= \frac{1}{2} E_{x\sim p} \frac{\partial (\log(p_\theta(x)/q(x)))^2}{\partial \theta} \\
\end{align}
$$

We used the fact that $$E_{x\sim p}\frac{\partial \log (p_\theta(x)) }{\partial \theta}=\sum_x \frac{\partial p_\theta(x)}{\partial \theta} = \frac{\partial}{\partial \theta}\sum_x p_\theta(x)=0$$
in the above derivation.

So $$f_X(x)=\frac{1}{2} \frac{\partial (\log(p_\theta(x)/q(x)))^2}{\partial \theta}$$ is an
unbiased estimator of $$\frac{\partial KL(p_\theta \Vert q)}{\partial \theta}$$. In fact, for any constant $$c$$,
$$\frac{1}{2} \frac{\partial (c+\log(p_\theta(x)/q(x)))^2}{\partial \theta}$$ is an unbiased estimator.

For estimator $f_N$, $$E_{x\sim p} \frac{\partial f_0(x)}{\theta}=0$$. So it is the worst possible estimator
we can get.

For $$f_S$$, we have

$$
\begin{align}
\frac{\partial f_S(x)}{\partial \theta}&=\frac{\partial \log(p_\theta(x)/q(x)}{\partial \theta} - \frac{q(x)}{p_\theta(x)}\frac{\partial \log(p_\theta(x)/q(x))}{\partial \theta} \\
&= (1 - q(x)/p_\theta(x))\frac{\partial \log(p_\theta(x)/q(x))}{\partial \theta} \\
&\approx -\log(q(x)/p(x)) \frac{\partial \log(p_\theta(x)/q(x))}{\partial \theta} \\
&= \log(p_\theta(x)/q(x)) \frac{\partial \log(p_\theta(x)/q(x))}{\partial \theta} \\
&= f_X(x) \\
\end{align}
$$

So when $$p(x)$$ is close to $$q(x)$$, $$\frac{\partial f_S(x)}{\partial \theta}$$
is a good approximation to the unbiased estimator $$\frac{\partial f_X(x)}{\partial \theta}$$.
This may explain why $$f_S$$ can still be useful in LLM RL finetuning (e.g. [DeepSeek R1 eq (2)](https://arxiv.org/abs/2501.12948)).
