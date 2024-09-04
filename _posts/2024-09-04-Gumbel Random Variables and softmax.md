

# Gumbel Random Variable

A Gumbel random variable G is a continuous random variable with cumulative distribution function $F(x) =  e^{ - e^{( - x /)}}$ ( can be generalized to shift and scale)  and probability density function $f(x) = e^{-x} e^{-e^{-x}}$  Mean of the Gumbel random variable is $\gamma \approx 0.5772$ also called the Euler-Mascheroni constant and standard deviation is $\pi / \sqrt{6} \approx 1.2825$

The probability distribution looks like:

<img src="https://github.com/apd10/any-paper-in-5-blog/blob/main/images/Gumbel.png" alt="drawing" style="width:200px;"/>

How to sample a Gumbel variable: Sample a U uniformly from [0,1] and then compute -log(-log(U)) ( via inverse transform sampling )


# Gumbel Reparameterization Trick
Let $(p_1, p_2, ..., p_n)$ be a set of non-negative coefficients. Let $g_1, g_2, ... g_n$ be independent gumbel variables. Then we have that 

$$\mathbf{P} (j = \textrm{arg} \max_i (g_i + \log p_i)) = \frac{p_i}{\sum_{i} p_i}$$

Equivalently,

$$\mathbf{P} (j = \textrm{arg} \max_i (g_i + p_i)) = \frac{e^{p_i}}{\sum_{i} e^{p_i}}$$

The above two equations can be used to sample from a categorical distribution. Also, it can be shown that,

$$\max_i \\{g_i + p_i\\} \sim \log \left( \sum_i e^{p_i}\right) + G$$

$$\mathbf{E} (\max_i \\{ g_i + p_i \\}) = \log \left( \sum_i e^{p_i}\right) + \gamma $$

The above equation can be used to estimate the partition function.


