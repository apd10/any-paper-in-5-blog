# Fast Inference from Transformers via Speculative Decoding

Observations: 
- The decoding step is memory bound, leaving compute unutilized
- The decoding is sequential. The amount of time to verify n-tokens vs. generate 1 token is in the similar ball park
- Not all next token predictions require complex models / long context understanding.

Methodology:
- Use a small model for $\gamma$ steps.
- Verify with one run of target model.
    - each token from left to right accept / reject.
    - once a token gets rejected, all the subsequent tokens are thrown
    - And one extra token is sampled from target model p(x) which is computed in this round.
- accept / reject the token using speculative sampling defined below where p(x)= target model distribution and q(x) = small model distribution


# Appendix
## Technical Details

### Speculative Sampling.
You want to sample from a distribution p(x). However the samples you get are from a distribution q(x). How do you ensure that you get the final sampling distribution as p(x). (Sound similar to rejection sampling?)

Algorithm: 
```
x' ~ q(x)
accept = True with probability min(1, p(x') / q(x'))
if accept:   # note that acceptance probability (a = \sum q(x') min(1, p(x') / q(x')) = \sum min(q(x'), p(x')))
    return x'
else:
    # resample from a new distribution
    x" ~ normalized(min(0, p(x) - q(x)))   # this probability is min(0, p(x) - q(x)) / (1 - a)
    return x"
```
Proof:
```
Pr(x') = Pr(x', accept) + Pr( reject, x')
       = q(x') min(1, p(x') / q(x')) + (1 - a) min(0, p(x') - q(x')) / (1-a)
       = min(q(x'), p(x')) + min(0, p(x') - q(x'))
       = p(x')
```

### Relation to Rejection Sampling.
The standard rejection sampling is 
```
x'~q(x)
r~U[0,1]
if r < p(x')/(q(x')M):
    return x'
else:
    start again.
```
The algorithm operates under the assumption that we have access to p(x) but we cannot sample from it. In our case of LLM, sampling from p(x) is possible but just to get the p(x) we need to run the bigger model. 
A modification of rejection sampling was proposed in paper appendix
```
x'~q(x)
r~U[0,1]
if r < p(x')/(q(x')M):
    return x'
else:
    x" ~ p(x)
    return x"
```

```
   Pr(x=x') = q(x') p(x') / q(x') M + (1 - a) p(x') = p(x') {1/M + (1 -a)}
```
The constant factor goes into normalization. The acceptance probability of this method is worse than speculative sampling recipe. (exercise)
