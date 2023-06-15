# Adaptive Proximal Algorithms for Convex Simple Bilevel optimization

This repository contains Julia code for the paper
[AdaBiM: An essentially adaptive proximal gradient method for convex simple bilevel optimization](https://arxiv.org/abs/2305.03559).

The problems that can be tackled are of the form 

$$
\begin{aligned}
    \text{minimize} \quad & f^1(x) + g^1(x) \\
    \text{subject to} \quad & x \in \arg\min_{w} f^2(w) + g^2(w)
\end{aligned}
$$

where $f^1,f^2$ are locally Lipschitz differentiable and $g^1,g^2$ are (possibly) nonsmooth prox-friendly functions. 

Algorithms are implemented [here](./adaptive_bilevel_algorithms.jl).

You can download the datasets required in some of the experiments by running:

```
julia --project=. download_datasets.jl
```

Numerical simulations for a few different problems are contained in subfolders.
For example, the linear inverse problem with the $\ell_1$ norm as the upper-level cost function can be found [here](https://github.com/pylat/adaptive-proximal-algorithms-bilevel-optimization/tree/master/experiments/logregNormL1). The `runme.jl` file includes the associated simulations. Executing `main()` will generate the plots in the same subfolder.
