---
layer: post
title: "Entities in Dynamic System"
author: "Wei Xu"
categories: [physics]
---

## Entities in Quantum Systems

A quantum system evolves according to the following equation:

$$
\frac{d\psi}{dt} = -iH\psi
$$

where $$H$$ is a Hermitian operator.

The question we want to ask is what can be considered as an entity in this system. To be
a little bit more specific, the question is what kind of $H$ dependent mathematical property
of $$\psi(t)$$ corresponds to an entity in the system at time $t$.

Hypothesis 1: If $$\psi(t)$$ can be approximated by a product state $$\psi_1(t) \otimes
\phi_1(t)$$ and the dimension of $$\psi_1(t)$$ is much smaller than that of $$\phi_1(t)$$,
and $\psi_1(t)$ can not be further approximated by a product state with similar goodness of
approximation, then $$\psi_1(t)$$ corresponds to some entity at time $$t$$. It is possible
to further approximate $$\psi_1(t)$$ by $$\psi_2(t)\otimes\phi_2(t)$$, but with a worse
approximation, then $$\psi_2(t)$$ can be considered as a sub-entity of $$\psi_1(t)$$. The goodness
of the approximation indicates how strong the interaction of the entity with the other part
of the system compared with its internal interations.

However, it is not always possible to approximate the a system using a product state. How
should we define entity in such systems?

## Tensor Product Spaces of Spins

Suppose the state space of our system is a tensor product space of $$N$$ spin systems, there are 
several interesting questions:

1. What kind of $$H$$ makes the system interesting?

2. How the system evolves starting from a pure product state?
