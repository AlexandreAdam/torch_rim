from abc import abstractmethod, ABC
from torch.nn import Module
from torch import Tensor


class Model(Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def initalize_hidden_states(self, dimensions:list[int], batch_size:int) -> Tensor:
        raise NotImplementedError("")
        

