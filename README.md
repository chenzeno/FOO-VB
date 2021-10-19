# FOO-VB
This is a pytorch implementation of FOO-VB algorithm for task agnostic continual learning. 

Please see our paper [Task Agnostic Continual Learning Using Online Variational Bayes with Fixed-Point Updates](https://arxiv.org/abs/2010.00373) for additional details.

# Discrete task-agnostic Permuted MNIST
We evaluate the algorithms on a task-agnostic scenario where the task boundaries are unknown. To do so, we use the Permuted MNIST benchmark for continual learning, but without informing the algorithms on task switches. Use the following command to run the experiment

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset permuted_mnist
```

# Continuous task-agnostic Permuted MNIST
We consider the case where the transition between tasks occurs gradually over time, so
the algorithm gets a mixture of samples from two 4 different tasks during the transition so the task boundaries are undefined. Use the following command to run the experiment
```
UDA_VISIBLE_DEVICES=0 python3 main.py --dataset continuous_permuted_mnist --alpha 0.6
```

# FOO-VB diagonal version 
The implementation code of the diagonal version of FOO-VB can be found in GitHub reposetory of our pre-print version [BGD](https://github.com/igolan/bgd).

# Reference
```
@article{10.1162/neco_a_01430,
    author = {Zeno, Chen and Golan, Itay and Hoffer, Elad and Soudry, Daniel},
    title = "{Task-Agnostic Continual Learning Using Online Variational Bayes With Fixed-Point Updates}",
    journal = {Neural Computation},
    volume = {33},
    number = {11},
    pages = {3139-3177},
    year = {2021},
    month = {10},
    abstract = "{Catastrophic forgetting is the notorious vulnerability of neural networks to the changes in the data distribution during learning. This phenomenon has long been considered a major obstacle for using learning agents in realistic continual learning settings. A large body of continual learning research assumes that task boundaries are known during training. However, only a few works consider scenarios in which task boundaries are unknown or not well defined: task-agnostic scenarios. The optimal Bayesian solution for this requires an intractable online Bayes update to the weights posterior.We aim to approximate the online Bayes update as accurately as possible. To do so, we derive novel fixed-point equations for the online variational Bayes optimization problem for multivariate gaussian parametric distributions. By iterating the posterior through these fixed-point equations, we obtain an algorithm (FOO-VB) for continual learning that can handle nonstationary data distribution using a fixed architecture and without using external memory (i.e., without access to previous data). We demonstrate that our method (FOO-VB) outperforms existing methods in task-agnostic scenarios. FOO-VB Pytorch implementation is available at https://github.com/chenzeno/FOO-VB.}",
    issn = {0899-7667},
    doi = {10.1162/neco_a_01430},
    url = {https://doi.org/10.1162/neco\_a\_01430},
    eprint = {https://direct.mit.edu/neco/article-pdf/33/11/3139/1966626/neco\_a\_01430.pdf},
}
```

# Requirements
* python = 3.6 
* pytorch = 1.2.0 
* torchvision = 0.4.0 
* numpy = 1.17.0
