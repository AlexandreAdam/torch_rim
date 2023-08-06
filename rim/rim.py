from typing import Callable

import torch
from torch import nn
from torch import Tensor
from torch.func import vjp
from .utils import load_architecture
from .definitions import  DEVICE
import numpy as np
from tqdm import tqdm
import time
import os, glob, re, json
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from .utils import NullEMA


class RIM(nn.Module):
    def __init__(
            self,
            dimensions:tuple,
            model:nn.Module=None,
            checkpoints_directory=None,
            score_fn:Callable=None,
            energy_fn:Callable=None,
            T:int=10,
            link_function:Callable=None,
            inverse_link_function:Callable=None,
            initialization_method:str="zeros",
            approximate_inverse_fn:Callable=None,
            score_preprocessing_method:str="Identity",
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999,
            device=DEVICE,
            **hyperparameters
            ):
        """
        This class implements a Recurrent Inference Machine (RIM) for parameter inference. 
        It takes in a model, dimensions of the parameter space, and various other optional
        parameters to customize the behavior of the RIM.
        
        Args:
            model (nn.Module): The model used for parameter inference.
            dimensions (tuple): The dimensions of the parameter space (excluding channels)
            score_fn (Callable, optional): The score function used for parameter inference. 
                If not provided, energy_fn must be provided.
            energy_fn (Callable, optional): The energy function used for parameter inference. 
                If not provided, score_fn must be provided.
            T (int, optional): The number of iterations for the RIM optimization. Defaults to 10.
            link_function (Callable, optional): The link function used to transform the parameters from model space to physical space.
                If provided, inverse_link_function must also be provided
            inverse_link_function (Callable, optional): The inverse of the link function used to transform the parameters from physical space to model space.
                If provided, link_function must also be provided.
            initialization_method (str, optional): The method used for parameter initialization. 
                Supported values are "zeros", "approximate_inverse", and "model". Defaults to "zeros".
            approximate_inverse_fn (Callable, optional): The function used to approximate the inverse of the link function. 
                Required if initialization_method is "approximate_inverse".
            score_preprocessing_method (str, optional): The method used for preprocessing the score. 
                Supported values are "ADAM", "RMSPROP", "arcsinh", and "identity". Defaults to "Identity".
            epsilon (float, optional): A small value used for numerical stability. Defaults to 1e-7.
        """
        super().__init__()
        if model is None and checkpoints_directory is None:
            raise ValueError("Must provide one of 'model' or 'checkpoints_directory'")
        if checkpoints_directory is not None:
            model, hyperparams = load_architecture(checkpoints_directory, model=model, device=device, hyperparameters=hyperparameters)
            hyperparameters.update(hyperparams) 
        else:
            try:
                hyperparameters.update(model.hyperparameters)
            except AttributeError:
                print("Model does not have an hyperparameters attribue")
        self.device = device
        self.model = model
        self.C = self.model.channels
        self.T = T
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        if link_function is not None:
            if inverse_link_function is None:
                raise ValueError("Both link function and its inverse must be provided")
            self.link_function = link_function
            self.inverse_link_function = inverse_link_function
        else:
            self.link_function = nn.Identity()
            self.inverse_link_function = nn.Identity()
        if initialization_method == "approximate_inverse":
            assert approximate_inverse_fn is not None, "approximate_inverse_fn must be provided"
            self.approximate_inverse_fn = approximate_inverse_fn
        if initialization_method not in ["zeros", "approximate_inverse", "model"]:
            raise ValueError("Initialization method not supported")
        self.initialization_method = initialization_method
        
        self.score_preprocessing_method = score_preprocessing_method
        if score_preprocessing_method.upper() == "ADAM":
            self.score_preprocessing = self.adam_score_update
        elif score_preprocessing_method.upper() == "RMSPROP":
            self.score_preprocessing = self.rmsprop_score_update
        elif score_preprocessing_method.lower() == "arcsinh":
            self.score_preprocessing = lambda x, t: torch.arcsinh(x)
        elif score_preprocessing_method.lower() == "identity":
            self.score_preprocessing = lambda x, t: x
        else:
            raise ValueError("score_preprocessing_method is not implemented")
        hyperparameters.update({
            "score_preprocessing_method": score_preprocessing_method,
            "initialization_method": initialization_method,
            "T": T,
            "epsilon": epsilon,
            "beta_1": beta_1,
            "beta_2": beta_2
            })
        self.hyperparameters = hyperparameters
        
        if score_fn is None:
            if energy_fn is None:
                raise ValueError("Either 'score_fn' or 'energy_fn' need to be provided")
            
            def score_fn(x, y, *args, **kwargs):
                _energy_fn = lambda x: self.energy_fn(x, y, *args, **kwargs)
                energy, vjp_func = vjp(_energy_fn, x)
                v = torch.ones_like(energy)
                return vjp_func(v)[0]
            
            def model_score_fn(x, y, *args, **kwargs):
                _energy_fn = lambda x: self.energy_fn(self.link_function(x), y, *args, **kwargs)
                energy, vjp_func = vjp(_energy_fn, x)
                v = torch.ones_like(energy)
                return vjp_func(v)[0]
            
        else:
            def model_score_fn(x, y, *args, **kwargs):
                linked_x, vjp_func = vjp(self.link_function, x)
                score = self.score_fn(linked_x, y, *args, **kwargs)
                return vjp_func(score)[0]
        self.score_fn = score_fn
        self.model_score_fn = model_score_fn
        self.energy_fn = energy_fn
    
    def initialization(self, observation) -> tuple[list[Tensor], Tensor, Tensor]:
        """
        From an observation, initialize the parameters to be inferred, x, and the
        hidden states of the recurrent neural net. 
        
        Args:
            observation (Tensor): The observation used for parameter initialization.
        
        Returns:
            tuple[list[Tensor], Tensor, Tensor]: A list for the optimization trajectories, 
                the initialized parameters x, and the initialized hidden states h.
        """
        batch_size = observation.shape[0]
        h = self.model.initalize_hidden_states(self.dimensions, batch_size).to(self.device)
        out = []
        if self.initialization_method == "zeros":
            x = torch.zeros((batch_size, self.C, *self.dimensions)).to(self.device)
        elif self.initialization_method == "approximate_inverse":
            x_param = self.approximate_inverse_fn(observation)
            x = self.inverse_link_function(x_param)
        elif self.initialization_method == "model":
            x = torch.zeros((batch_size, self.C, *self.dimensions)).to(self.device)
            score = torch.zeros((batch_size, self.C, *self.dimensions)).to(self.device)
            x, h = self.model(x, score, observation, h)
            out.append(x)
        else:
            raise ValueError("Initialization method not supported")
        if self.score_preprocessing_method.lower() in ["adam", "rmsprop"]:
            self._score_mean = torch.zeros_like(x)
            self._score_var = torch.zeros_like(x)
        return out, x, h
    

    def forward(self, y: Tensor, *args, **kwargs) -> list[Tensor]:
        """
        Perform the forward pass of the RIM optimization.
        
        Args:
            y (Tensor): The observation used for the optimization.
            args and kwargs: Additional arguments and keyword arguments for the score function.
        
        Returns:
            list[Tensor]: The RIM optimization trajectories, represented as a list of parameter x at every iteration of the recurrent series.
        """
        out, x, h = self.initialization(y)
        for t in range(self.T):
            with torch.no_grad():
                score = self.model_score_fn(x, y, *args, **kwargs)
                score = self.score_preprocessing(score, t)
            g, h = self.model(x, y, score, h)
            x = x + g
            out.append(x)
        return out
    
    def predict(self, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Perform a prediction using the RIM optimization.
        
        Args:
            y (Tensor): The observation used for the prediction.
            args and kwargs: Additional arguments and keyword arguments for the score function.
        
        Returns:
            Tensor: The predicted parameter x after the RIM optimization (in physical space). 
        """
        return self.link_function(self.forward(y, *args, **kwargs)[-1])
            
    def adam_score_update(self, score: Tensor, time_step: int):
        """
        Update the score using the ADAM algorithm.
        
        Args:
            score (Tensor): The score to be updated.
            time_step (int): The current time step of the optimization.
        
        Returns:
            Tensor: The updated score.
        """
        self._score_mean = self.beta_1 * self._score_mean + (1 - self.beta_1) * score
        self._score_var = self.beta_2 * self._score_var + (1 - self.beta_2) * score**2
        m_hat = self._score_mean / (1 - self.beta_1 ** (time_step + 1))
        v_hat = self._score_var / (1 - self.beta_2 ** (time_step + 1))
        return m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def rmsprop_score_update(self, score: Tensor, time_step: int):
        """
        Update the score using the RMSProp algorithm.
        
        Args:
            score (Tensor): The score to be updated.
            time_step (int): The current time step of the optimization.
        
        Returns:
            Tensor: The updated score.
        """
        self._score_mean = self.beta_1 * self._score_mean + (1 - self.beta_1) * score
        v_hat = self._score_var / (1 - self.beta_2 ** (time_step + 1))
        return score / (torch.sqrt(v_hat) + self.epsilon)
    
    def loss_fn(self, y: Tensor, x_true: Tensor, *args, **kwargs) -> Tensor:
        B = y.shape[0]
        # if w is None:
        w = torch.ones_like(x_true) #TODO support weights
        x_series = self(y, *args, **kwargs)
        x_true = self.inverse_link_function(x_true) # important to compute the loss in model space
        loss = sum([(w * (x - x_true)**2).sum() for x in x_series]) / B
        return loss

    def fit(
        self,
        dataset,
        epochs=100,
        learning_rate=1e-4,
        scheduler=None,
        batch_size=1,
        shuffle=False,
        patience=float('inf'),
        tolerance=0,
        max_time=float('inf'),
        warmup=0,
        clip=0.,
        checkpoints_directory=None,
        model_checkpoint=None,
        checkpoints=10,
        models_to_keep=3,
        ema_decay=0,
        seed=None,
        logname=None,
        logdir=None,
        n_iterations_in_epoch=None,
        logname_prefix="rim",
        verbose=0
    ):
        """
        Train the model on the provided dataset.

        Parameters:
            dataset (torch.utils.data.Dataset): The training dataset.
            learning_rate (float, optional): The learning rate for optimizer. Default is 1e-4.
            batch_size (int, optional): The batch size for training. Default is 1.
            shuffle (bool, optional): Whether to shuffle the dataset during training. Default is False.
            epochs (int, optional): The number of epochs for training. Default is 100.
            patience (float, optional): The patience value for early stopping. Default is infinity.
            tolerance (float, optional): The tolerance value for early stopping. Default is 0.
            max_time (float, optional): The maximum training time in hours. Default is infinity.
            warmup (int, optional): The number of warmup iterations for learning rate. Default is 0.
            clip (float, optional): The gradient clipping value. Default is 0.
            model_checkpoint (float, optional): If checkpoints_directory is provided, this can be used to restart training from checkpoint.
            checkpoints_directory (str, optional): The directory to save model checkpoints. Default is None.
            checkpoints (int, optional): The interval for saving model checkpoints. Default is 10 epochs.
            models_to_keep (int, optional): The number of best models to keep. Default is 3.
            seed (int, optional): The random seed for numpy and torch. Default is None.
            logname (str, optional): The logname for saving checkpoints. Default is None.
            logdir (str, optional): The path to the directory in which to create the new checkpoint_directory with logname.
            logname_prefix (str, optional): The prefix for the logname. Default is "score_model".

        Returns:
            list: List of loss values during training.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if scheduler is not None: # assumes the argument of the sceduler were wrapped in before
            scheduler = scheduler(optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)

        ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay) if ema_decay > 0 else NullEMA() 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        if n_iterations_in_epoch is None:
            n_iterations_in_epoch = len(dataloader)
        
        hyperparameters = self.hyperparameters
        # ==== Take care of where to write checkpoints and stuff =================================================================
        if checkpoints_directory is not None:
            if os.path.isdir(checkpoints_directory):
                logname = os.path.split(checkpoints_directory)[-1]
        elif logname is None:
            logname = logname_prefix + "_" + datetime.now().strftime("%y%m%d%H%M%S")

        save_checkpoint = False
        latest_checkpoint = 0
        if checkpoints_directory is not None or logdir is not None:
            save_checkpoint = True
            if checkpoints_directory is None: # the way to create a new directory is using logdir
                checkpoints_directory = os.path.join(logdir, logname)
            if not os.path.isdir(checkpoints_directory):
                os.mkdir(checkpoints_directory)
                with open(os.path.join(checkpoints_directory, "script_params.json"), "w") as f:
                    json.dump(
                        {
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "shuffle": shuffle,
                            "epochs": epochs,
                            "patience": patience,
                            "tolerance": tolerance,
                            "max_time": max_time,
                            "warmup": warmup,
                            "clip": clip,
                            "checkpoint_directory": checkpoints_directory,
                            "checkpoints": checkpoints,
                            "models_to_keep": models_to_keep,
                            "ema_decay": ema_decay,
                            "seed": seed,
                            "logname": logname,
                            "logname_prefix": logname_prefix,
                        },
                        f,
                        indent=4
                    )
                with open(os.path.join(checkpoints_directory, "model_hparams.json"), "w") as f:
                    json.dump(hyperparameters, f, indent=4)

            # ======= Load checkpoints if they are provided ===============================================================
            paths = glob.glob(os.path.join(checkpoints_directory, "checkpoint*.pt"))
            opt_paths = glob.glob(os.path.join(checkpoints_directory, "optimizer*.pt"))
            checkpoint_indices = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
            scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
            if checkpoint_indices:
                if model_checkpoint is not None:
                    checkpoint_path = paths[checkpoint_indices.index(model_checkpoint)]
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.model.device))
                    optimizer.load_state_dict(torch.load(opt_paths[checkpoints == model_checkpoint], map_location=self.device))
                    print(f"Loaded checkpoint {model_checkpoint} of {logname}")
                    latest_checkpoint = model_checkpoint
                else:
                    max_checkpoint_index = np.argmax(checkpoint_indices)
                    checkpoint_path = paths[max_checkpoint_index]
                    opt_path = opt_paths[max_checkpoint_index]
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                    print(f"Loaded checkpoint {checkpoint_indices[max_checkpoint_index]} of {logname}")
                    latest_checkpoint = checkpoint_indices[max_checkpoint_index]

        if seed is not None:
            torch.manual_seed(seed)
        best_loss = float('inf')
        losses = []
        step = 0
        global_start = time.time()
        estimated_time_for_epoch = 0
        out_of_time = False

        for epoch in (pbar := tqdm(range(epochs))):
            if (time.time() - global_start) > max_time * 3600 - estimated_time_for_epoch:
                break
            epoch_start = time.time()
            time_per_step_epoch_mean = 0
            cost = 0
            data_iter = iter(dataloader)
            for _ in range(n_iterations_in_epoch):
                start = time.time()
                try:
                    y, x_true, *args = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    y, x_true, *args = next(data_iter)
                optimizer.zero_grad()
                loss = self.loss_fn(y, x_true, *args)
                loss.backward()

                if step < warmup:
                    for g in optimizer.param_groups:
                        g['lr'] = learning_rate * np.minimum(step / warmup, 1.0)

                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)

                optimizer.step()
                ema.update()

                _time = time.time() - start
                time_per_step_epoch_mean += _time
                cost += float(loss)
                step += 1
            scheduler.step()

            time_per_step_epoch_mean /= len(dataloader)
            cost /= len(dataloader)
            pbar.set_description(f"Epoch {epoch + 1:d} | Cost: {cost:.1e} |")
            losses.append(cost)
            if verbose >= 2:
                print(f"epoch {epoch} | cost {cost:.2e} | time per step {time_per_step_epoch_mean:.2e} s")
            elif verbose == 1:
                if (epoch + 1) % checkpoints == 0:
                    print(f"epoch {epoch} | cost {cost:.1e}")

            if np.isnan(cost):
                print("Model exploded and returns NaN")
                break

            if cost < (1 - tolerance) * best_loss:
                best_loss = cost
                patience = patience
            else:
                patience -= 1

            if (time.time() - global_start) > max_time * 3600:
                out_of_time = True

            if save_checkpoint:
                if (epoch + 1) % checkpoints == 0 or patience == 0 or epoch == epochs - 1 or out_of_time:
                    latest_checkpoint += 1
                    with open(os.path.join(checkpoints_directory, "score_sheet.txt"), mode="a") as f:
                        f.write(f"{latest_checkpoint} {cost}\n")
                    with ema.average_parameters():
                        torch.save(self.model.state_dict(), os.path.join(checkpoints_directory, f"checkpoint_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoints_directory, f"optimizer_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    paths = glob.glob(os.path.join(checkpoints_directory, "*.pt"))
                    checkpoint_indices = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
                    scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
                    if len(checkpoint_indices) > 2*models_to_keep: # has to be twice since we also save optimizer states
                        index_to_delete = np.argmin(checkpoint_indices)
                        os.remove(os.path.join(checkpoints_directory, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
                        os.remove(os.path.join(checkpoints_directory, f"optimizer_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
                        del scores[index_to_delete]
                        del checkpoint_indices[index_to_delete]

            if patience == 0:
                print("Reached patience")
                break

            if out_of_time:
                print("Out of time")
                break

            if epoch > 0:
                estimated_time_for_epoch = time.time() - epoch_start

        ema.copy_to(self.model.parameters())
        print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
        return losses
