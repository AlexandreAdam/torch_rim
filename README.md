# RIM: Recurrent Inference Machines
[![PyPI version](https://badge.fury.io/py/torch_rim.svg)](https://badge.fury.io/py/torch_rim)
[![codecov](https://codecov.io/gh/AlexandreAdam/torch_rim/branch/master/graph/badge.svg)](https://codecov.io/gh/AlexandreAdam/torch_rim)

This is an implementation of a Recurrent Inference Machine (see [Putzky & Welling (2017)](https://arxiv.org/abs/1706.04008)) 
alongside some standard neural network architectures for the type of problem RIM can solve.

# Installation
To install the package, you can use pip:

```bash
pip install torch_rim
```
# Usage

# Background

A RIM is a meta-learner. It is trained not as a feed-forward neural network, but rather as an optimisation 
algorithm. More specifically, the RIM is given a *problem instance* specified by a likelihood score $\nabla_\mathbf{x} \log p(y \mid x)$, 
or more generally a posterior scoe function $\nabla_{\mathbf{x}} \log p(x \mid y)$, and an observation $y$ to condition said posterior. 

The RIM uses this information to perform a learned gradient ascent algorithm on the posterior. 
This procedure will produce a MAP estimate of the parameters of interests $\mathbf{x}$ when the RIM is trained
```math
\begin{align}
\hat{\mathbf{x}}_{t+1} &= \hat{\mathbf{x}}_t + \mathbf{g}_\theta (\hat{\mathbf{x}}_t, y, \nabla_{\hat{\mathbf{x}}_t} \log p(\hat{\mathbf{x}}_t, \mathbf{y}), \mathbf{h}_t) \hspace{0.5cm} t \in [0, T]\\
\mathbf{h}_{t+1} &= g_\theta(\hat{\mathbf{x}}_t, \mathbf{y}, \nabla_{\hat{\mathbf{x}}_t} \log p(\hat{\mathbf{x}}_t, \mathbf{y}), \mathbf{h}_t)
\end{align}
```
In the last equation, $g_\theta$ is a neural network that act as the *gradient* in the gradient ascent algorithm. The second equation represent an hidden state 
much like modern optimisation algorithm like ADAM ([Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980)) or RMSProp 
([Hinton (2011)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)) that uses hidden state to aggregate information 
about the trajectory of the particle during optimisation. In this case, $\mathbf{h}$ is the hidden state of a Gated Recurrent Unit 
([Chung et al. (2014)](https://arxiv.org/abs/1412.3555)).


The RIM is trained using an outer loop optimisation contructed based on labels $\mathbf{x}$ and a simulator $F$: 
```math
$\mathbf{y} = F(\mathbf{x}) + \boldsymbol{\eta}$. 
```
Equipped with a dataset of problem instances and their solutions 
```math
\mathcal{D} = \{(\mathbf{y}, \mathbf{x}, \nabla_\mathbf{x}\log p(\mathbf{x} \mid \mathbf{y})\}$
```
we can then use train the RIM to make it's gradient ascent trajectories as efficient as possible by minimizing the
following meta-learning loss
```math
\mathcal{L}_\theta = \mathbb{E}_\mathcal{D} \left[ \sum_{t=1}^T\lvert \mathbf{w}_t(\hat{\mathbf{x}}_t - \mahtbf{x}\rVert^2_2 \right]
```
where $\mathbf{w}_t$ weighs each MSE of the trajectory.


# Citations
If you find this package useful, consider citing
Putzky

You might also find some inspiration in those applications
cite morningstar, me,etc. 
