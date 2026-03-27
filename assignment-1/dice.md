---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
import scipy
import matplotlib.pyplot as plt
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Dice

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Throw the provided dice more than 100 times. Count how many times each face apeared in the results and write it down in array

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
data = np.asarray([18,16,14,18,18,16])
```

```{code-cell} ipython3
faces = np.arange(1, 7)
plt.bar(faces, data)
plt.xlabel('Dice face')
plt.ylabel('Frequency')
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

First number in the array should be the number of ones in the results and last one should be the number of sixes.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Plot the results using the [`matplotlib.pyplot.bar`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html) function.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Assuming an uniform prior, what is the posterior for the probability of getting a six? Plot this function.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Provide the MAP, mean and median of the distribution. Please print them out and mark them on the plot.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["hint"]}

__Hint__ For median use the `isf` function of the distribution from `scipy.stats`.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Estimate the HDR containing the 90% probability. Use the `bda.stats.hdr_f` function.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import bda 
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
help(bda.stats.hdr_f)
```

```{code-cell} ipython3
from scipy.stats import beta

n = np.sum(data)
k = data[5]

alpha_post = 1 + k
beta_post = 1 + n - k
dist = beta(alpha_post, beta_post)

map_val = (alpha_post - 1) / (alpha_post + beta_post - 2)
mean_val = dist.mean()
median_val = dist.isf(0.5)

print(f"MAP: {map_val}")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")

x = np.linspace(0, 1, 1000)
y = dist.pdf(x)

plt.plot(x, y)
plt.axvline(map_val, color='r')
plt.axvline(mean_val, color='g')
plt.axvline(median_val, color='k')
plt.xlabel('Probability of six')
plt.ylabel('Density')
plt.show()

hdr = bda.stats.hdr_f(dist.pdf, 0.90, a=1e-5, b=1-1e-5)
print(f"90% HDR: {hdr}")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

 In case of numerical problems you may need to constrain the (a,b) interval.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the following but this time for the prior use Beta distribution with mean equal to 1/6 and standard deviation equal to 1/100.

```{code-cell} ipython3
mu = 1/6
sigma = 1/100

nu = (mu * (1 - mu)) / (sigma**2) - 1
alpha_prior = mu * nu
beta_prior = (1 - mu) * nu

alpha_post2 = alpha_prior + k
beta_post2 = beta_prior + n - k
dist2 = beta(alpha_post2, beta_post2)

map_val2 = (alpha_post2 - 1) / (alpha_post2 + beta_post2 - 2)
mean_val2 = dist2.mean()
median_val2 = dist2.isf(0.5)

print(f"MAP: {map_val2}")
print(f"Mean: {mean_val2}")
print(f"Median: {median_val2}")

x2 = np.linspace(0, 1, 1000)
y2 = dist2.pdf(x2)

plt.plot(x2, y2)
plt.axvline(map_val2, color='r')
plt.axvline(mean_val2, color='g')
plt.axvline(median_val2, color='k')
plt.xlabel('Probability of six')
plt.ylabel('Density')
plt.show()

hdr2 = bda.stats.hdr_f(dist2.pdf, 0.90, a=0.1, b=0.3)
print(f"90% HDR: {hdr2}")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 3

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Let's suppose that the dice is crooked and

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$p_1=\frac{1}{6}-\epsilon,\, p_2=p_3=p_4=p_5=
\frac{1}{6},\, p_6 =\frac{1}{6}+\epsilon$$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

with $\epsilon=0.01$. Assuming uniform prior estimate  how many throws are needed to ascertain that a dice is crooked with 90% accuracy.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["hint"]}

__Hint__: This is a binomial distribution with $p=\frac{1}{6}+\epsilon$. Assume that for large $n$ the number of sixes is equal to $p\cdot n$. Calculate the posterior and 90% HDR. By trial and error estimate smallest $n$ such that $p=\frac{1}{6}$ lies outside of the HDR.

```{code-cell} ipython3
p_true = 1/6 + 0.01
p_fair = 1/6

n_test = 1
while True:
    k_test = p_true * n_test
    dist_test = beta(1 + k_test, 1 + n_test - k_test)
    
    mu = dist_test.mean()
    std = dist_test.std()
    a_dyn = max(1e-5, mu - 5 * std)
    b_dyn = min(1 - 1e-5, mu + 5 * std)
    
    hdr_test = bda.stats.hdr_f(dist_test.pdf, 0.90, a=a_dyn, b=b_dyn)
    
    if hdr_test[0][0] > p_fair:
        print(f"Uniform prior minimum n: {n_test}")
        break
    n_test += 1

n_test2 = 1
while True:
    k_test2 = p_true * n_test2
    dist_test2 = beta(alpha_prior + k_test2, beta_prior + n_test2 - k_test2)
    
    mu2 = dist_test2.mean()
    std2 = dist_test2.std()
    a_dyn2 = max(1e-5, mu2 - 5 * std2)
    b_dyn2 = min(1 - 1e-5, mu2 + 5 * std2)
    
    hdr_test2 = bda.stats.hdr_f(dist_test2.pdf, 0.90, a=a_dyn2, b=b_dyn2)
    
    if hdr_test2[0][0] > p_fair:
        print(f"Informative prior minimum n: {n_test2}")
        break
    n_test2 += 1
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Perform same calculations for the prior from Problem 2.
