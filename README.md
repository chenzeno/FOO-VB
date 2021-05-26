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


# Requirements
* python = 3.6 
* pytorch = 1.2.0 
* torchvision = 0.4.0 
* numpy = 1.17.0
