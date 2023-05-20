# Adaptive Proximal Algorithms

This repository contains Julia code for the paper
[AdaBiM: An adaptive proximal gradient method for structured
convex bilevel optimization](https://arxiv.org/pdf/2305.03559.pdf).

Algorithms are implemented [here](./adaptive_bilevel_algorithms.jl).

You can download the datasets required in some of the experiments by running:

```
julia --project=. download_datasets.jl
```

Experiments on a few different problems are contained in subfolders.
For example, run the linear inverse problem experiments as follows:

```
julia --project=. experiments/linear_inverse/runme.jl
```

This will generate plots in the same subfolder.
