=======================================
RIM: Recurrent Inference Machines
=======================================

.. image:: https://badge.fury.io/py/torch_rim.svg
    :target: https://badge.fury.io/py/torch_rim

.. image:: https://codecov.io/gh/AlexandreAdam/torch_rim/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/AlexandreAdam/torch_rim

This is an implementation of a Recurrent Inference Machine (see `Putzky & Welling (2017) <https://arxiv.org/abs/1706.04008>`_)
alongside some standard neural network architectures for the type of problem RIM can solve.

Installation
------------

To install the package, you can use pip:

.. code-block:: bash

    pip install torch-rim

Usage
-----

.. code-block:: python

    from torch_rim import RIM, Hourglass, Unet
    from torch.func import vmap

    # B is the batch size
    # C is the input channels
    # dimensions are the spatial dimensions (e.g. [28, 28] for MNIST)

    # Create a score_fn, e.g. a Gaussian likelihood score function
    @vmap
    def score_fn(x, y, A, Sigma): # must respect the signature (x, y, *args)
        # A is a linear forward model, Sigma is the noise covariance
        return (y - A @ x) @ Sigma.inverse() @ A

    # ... or a Gaussian energy function (unnormalized log probability)
    @vmap
    def energy_fn(x, y, F, Sigma):
        # F is a general forward model
        return (y - F(x)) @ Sigma.inverse() @ (y - F(x))

    # Create a RIM instance with the Hourglass neural network back-bone and the score function
    net = Hourglass(C, dimensions=len(dimensions))
    rim = RIM(dimensions, net, score_fn=score_fn)

    # ... or with the energy function
    rim = RIM(dimensions, net, energy_fn=energy_fn)

    # Train the rim, and save its weight in checkpoints_directory
    model.fit(dataset=your_dataset, epochs=100, learning_rate=1e-4, checkpoints_directory=checkpoints_directory)

    # Make a prediction on an observation y
    x_hat = rim.predict(y, A, Sigma) # of with the signature (y, F, Sigma) with the energy_fn


Background
----------
A RIM is a gradient-based meta-learning algorithm. It is trained not as a feed-forward neural network, but rather as an optimisation 
algorithm. More specifically, the RIM is given a *problem instance* specified by a likelihood score :math:`\nabla_\mathbf{x} \log p(y \mid x)`, 
or more generally a posterior scoe function :math:`\nabla_{\mathbf{x} \mid \mathbf{y}} \equiv \nabla_{\mathbf{x}} \log p(x \mid y)`, and an observation :math:`y` to condition said posterior.

The RIM uses this information to perform a learned gradient ascent algorithm on the posterior. This procedure will produce a MAP estimate of the parameters of interests :math:`\mathbf{x}` when the RIM is trained.

.. math::

   \hat{\mathbf{x}}_{t+1} &= \hat{\mathbf{x}}_t + \mathbf{g}_\theta (\hat{\mathbf{x}}_t,\, \mathbf{y},\, \nabla_{\hat{\mathbf{x}}_t \mid \mathbf{y}},\, \mathbf{h}_t)\\
   \mathbf{h}_{t+1} &= \mathbf{g}_\theta(\hat{\mathbf{x}}_t,\, \mathbf{y},\, \nabla_{\hat{\mathbf{x}}_t \mid \mathbf{y}},\, \mathbf{h}_t)

for :math:`t \in [0, T]`. 

In the last equation, :math:`\mathbf{g}_\theta` is a neural network that act as the *gradient* in the gradient ascent algorithm. The second equation represent an hidden state much like modern optimisation algorithm like ADAM (`Kingma & Ba (2014) <https://arxiv.org/abs/1412.6980>`_) or RMSProp (`Hinton (2011) <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_) that uses hidden state to aggregate information about the trajectory of the particle during optimisation. In this case, :math:`\mathbf{h}` is the hidden state of a Gated Recurrent Unit (`Chung et al. (2014) <https://arxiv.org/abs/1412.3555>`_).

The RIM is trained using an outer loop optimisation contructed with labels :math:`\mathbf{x}` (the parameters of interests) and a simulator :math:`F`: 

.. math::

   \mathbf{y} = F(\mathbf{x}) + \boldsymbol{\eta}

Equipped with a dataset of problem instances and their solutions :math:`\mathbf{x}`

.. math::

   \mathcal{D} = \big\{\mathbf{y}^{(i)},\, \mathbf{x}^{(i)},\, \nabla_{\mathbf{x} \mid \mathbf{y}}^{(i)}\big\}_{i=1}^N

we can train the RIM to make it's gradient ascent trajectories as efficient as possible by minimizing a weighted mean squared loss

.. math::

   \mathcal{L}_\theta = \mathbb{E}_\mathcal{D} \left[ \sum_{t=1}^T\lVert \mathbf{w}_t(\hat{\mathbf{x}}_t - \mathbf{x}\rVert^2_2 \right]

:math:`\mathbf{w}_t` weighs each MSE on the parameter trajectory :math:`\big\{\mathbf{x}_t\big\}_{t=0}^T`.


Citations
---------

If you find this work useful, please consider citing:

- `Putzky & Welling (2017) <http://arxiv.org/abs/1706.04008>`_
    .. code-block:: bibtex

    @article{Putzky2017,
        author = {Putzky, Patrick and Welling, Max},
        journal = {arXiv e-prints},
        archivePrefix = {arXiv},
        eprint = {0902.0885},
        primaryClass = {Neural and Evolutionary Computing},
        title = {{Recurrent Inference Machines for Solving Inverse Problems}},
        url = {http://arxiv.org/abs/1706.04008},
        year = {2017}
    }

- `Adam et al. (2023) <https://iopscience.iop.org/article/10.3847/1538-4357/accf84#references>`_ (Gravitational lensing)
    .. code-block:: bibtex

    @article{Adam2023,
           author = {{Adam}, Alexandre and {Perreault-Levasseur}, Laurence and {Hezaveh}, Yashar and {Welling}, Max},
            title = "{Pixelated Reconstruction of Foreground Density and Background Surface Brightness in Gravitational Lensing Systems Using Recurrent Inference Machines}",
         keywords = {Convolutional neural networks, Astronomical simulations, Nonparametric inference},
          journal = {\apj},
             year = 2023,
            month = jul,
           volume = {951},
           number = {1},
              eid = {6},
            pages = {6},
              doi = {10.3847/1538-4357/accf84},
    archivePrefix = {arXiv},
           eprint = {2301.04168},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...951....6A},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

- `Morningstar et al. (2019) <https://iopscience.iop.org/article/10.3847/1538-4357/ab35d7>`_ (Gravitational lensing)
    .. code-block:: bibtex

    @article{Morningstar2019,
           author = {{Morningstar}, Warren R. and {Perreault Levasseur}, Laurence and {Hezaveh}, Yashar D. and {Blandford}, Roger and {Marshall}, Phil and {Putzky}, Patrick and {Rueter}, Thomas D. and {Wechsler}, Risa and {Welling}, Max},
            title = "{Data-driven Reconstruction of Gravitationally Lensed Galaxies Using Recurrent Inference Machines}",
          journal = {\apj},
         keywords = {Convolutional neural networks, Strong gravitational lensing, Neural networks, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
             year = 2019,
            month = sep,
           volume = {883},
           number = {1},
              eid = {14},
            pages = {14},
              doi = {10.3847/1538-4357/ab35d7},
    archivePrefix = {arXiv},
           eprint = {1901.01359},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJ...883...14M},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

- `Morningstar et al. (2018) <https://arxiv.org/abs/1808.00011>`_ (Interferometric imaging)
    .. code-block:: bibtex
    @article{Morningstar2018,
           author = {{Morningstar}, Warren R. and {Hezaveh}, Yashar D. and {Perreault Levasseur}, Laurence and {Blandford}, Roger D. and {Marshall}, Philip J. and {Putzky}, Patrick and {Wechsler}, Risa H.},
            title = "{Analyzing interferometric observations of strong gravitational lenses with recurrent and convolutional neural networks}",
          journal = {arXiv e-prints},
         keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics},
             year = 2018,
            month = jul,
              eid = {arXiv:1808.00011},
            pages = {arXiv:1808.00011},
              doi = {10.48550/arXiv.1808.00011},
    archivePrefix = {arXiv},
           eprint = {1808.00011},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2018arXiv180800011M},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

- `Lönning et al. (2019) <https://pubmed.ncbi.nlm.nih.gov/30703579>`_ (MRI imaging)
    .. code-block:: bibtex
    @article{Lonning2019,
        author = {L{\o}nning, Kai and Putzky, Patrick and Sonke, Jan Jakob and Reneman, Liesbeth and Caan, Matthan W.A. and Welling, Max},
        doi = {10.1016/j.media.2019.01.005},
        issn = {13618423},
        journal = {Medical Image Analysis},
        keywords = {Deep learning,Inverse problems,MRI,Reconstruction},
        month = {apr},
        pages = {64--78},
        pmid = {30703579},
        publisher = {Elsevier},
        title = {{Recurrent inference machines for reconstructing heterogeneous MRI data}},
        volume = {53},
        year = {2019}
    }

- `Modi et al. (2021) <https://arxiv.org/abs/2104.12864>`_ (Initial conditions of the Universe)
    .. code-block:: bibtex
    @article{Modi2021,
           author = {{Modi}, Chirag and {Lanusse}, Fran{\c{c}}ois and {Seljak}, Uro{\v{s}} and {Spergel}, David N. and {Perreault-Levasseur}, Laurence},
            title = "{CosmicRIM : Reconstructing Early Universe by Combining Differentiable Simulations with Recurrent Inference Machines}",
          journal = {arXiv e-prints},
    archivePrefix = {arXiv},
           eprint = {2104.12864},
     primaryClass = {astro-ph.CO},
         keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
             year = 2021,
            month = apr,
              eid = {arXiv:2104.12864},
              doi = {10.48550/arXiv.2104.12864},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210412864M},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


