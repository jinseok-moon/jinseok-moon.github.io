---
title: "Online softmax"
slug: "online-softmax"
date: "2025-07-12"
draft: false
categories:
  - cuda
showToc: true
---

## Original softmax

$$
\sigma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum^K_{j=1}e^{z_j}}
$$

![](images/image.png)

The formula for softmax is shown above. For an input vector, a naïve implementation requires two loads and one store, for a total of three memory accesses per element. Softmax converts raw scores into a probability distribution over the input values. It is extremely useful, but in practice it is numerically fragile due to floating‑point limitations. Floating point represents real numbers within a finite dynamic range. Because softmax uses the exponential function $e^z$, the values can grow very largeand the sum in the denominator is prone to overflow. In the opposite direction, for large negative inputs $e^z$ becomes very close to zeroand the denominator can underflow toward 0, leading to undefined behavior when used in division.

## Safe softmax
To address this, we can rewrite the formula by subtracting the maximum value from every element. After this shift, all elements are less than or equal to zero and the dynamic range is much narrower. In the denominator, one of the terms becomes $e^0 = 1$ (the term corresponding to the maximum element), which guarantees that the denominator cannot be exactly zero.

$$
	\sigma_i(\mathbf{z}) = \frac{e^{z_i - \max(z)}}{\sum^K_{j=1}e^{z_j-\max(z)}}
$$

![](images/image-1.png)

This "safe" softmax is mathematically equivalent to the original definition but avoids overflow and underflow in practice, so it is the version that most implementations use. The downside is that we now need an extra pass to compute the maximum $m_k$, so the number of memory accesses increases to four.

## Online softmax calculation
`Online softmax` is a way to reduce the number of memory accesses again. If you look closely at the safe softmax derivation, you will notice that we do not strictly need the *global* maximum of the vector; we just need a running value that is large enough to keep the exponentials well behaved. We can maintain such a value using local maxima as we stream through the data.

![](images/image-2.png)

The denominator $d_V$ in the algorithm above can be expressed recursively. Using the exponential rules, we can factor out the terms of the form $e^m$.

![](images/image-3.png)

This derivation shows how to update the denominator $d_V$ on the fly while tracking only a local maximum. Is this still numerically safe? Yes: for each step $j$, the stabilizing term $m_j$ is chosen large enough that $x_i - m_j \le 0$ for all processed elements, so $e^{x_i - m_j} \le 1$. As a result, $d_j$ stays within a well‑behaved range between 1 and $j$, preventing overflow or underflow.

## References
1. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
