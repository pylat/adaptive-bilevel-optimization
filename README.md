# Adaptive Proximal Algorithms for Convex Bilevel optimization

This repository contains Julia code for the paper
[AdaBiM: An adaptive proximal gradient method for structured
convex bilevel optimization](https://arxiv.org/pdf/2305.03559.pdf).

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

Experiments on a few different problems are contained in subfolders.
For example, run the linear inverse problem with the $\ell_1$ norm as the upper level cost as follows:

```
julia --project=. experiments/LinearInverse/NormL1/runme.jl
```

This will generate plots in the same subfolder.