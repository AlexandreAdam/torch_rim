from typing import Callable

import torch
from torch import nn
from torch import Tensor
from torch.func import vjp
from .definitions import DEVICE


class RIM(nn.Module):
    def __init__(
            self,
            model:nn.Module,
            dimensions:tuple,
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
            device=DEVICE
            ):
        """
        Recurrent Inference Machine
        
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
            link_function (Callable, optional): The link function used to transform the parameters. 
                If provided, inverse_link_function must also be provided.
            inverse_link_function (Callable, optional): The inverse of the link function used to transform the parameters. 
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
            Tensor: The predicted parameter x after the RIM optimization (in parameter space). 
        """
        return self.inverse_link_function(self.forward(y)[-1])
            
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

    
    def loss_fn(y: Tensor, x_true: Tensor, w: Tensor = None, *args, **kwargs) -> Tensor:
        B = y.shape[0]
        if w is None:
            w = torch.ones_like(x_true)
        x_series = self(y, *args, **kwargs)
        loss = sum([(w * (x - x_true)**2).sum() for x in x_series]) / B
        return loss

    # def fit(): #TODO import code from score model here and adapt
