import torch
from rim import RIM, Hourglass
import os, shutil

def setup(dim):
    H = 32
    B = 10
    C = 15 # channels
    T = 10
    ch_mult = (2, 2)
    dim = 1
    D = [16]*dim
    x = torch.randn(B, C, *D)
    y = torch.randn_like(x)
    model = Hourglass(C, ch_mult=ch_mult, dimensions=dim)
    score_fn = lambda x, y: torch.randn_like(x)
    energy_fn = lambda x, y: torch.ones([B])
    return x, y, model, score_fn, energy_fn, H, B, C, T, D


def test_training():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(D, model, score_fn=score_fn, T=T)
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2*B
        def __getitem__(self, index):
            return torch.randn(C, *D), torch.randn(C, *D)
    
    dataset = Dataset()
    # Set the hyperparameters and other options for training
    checkpoints_directory = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"
    learning_rate = 1e-3
    batch_size = B
    epochs = 10
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    seed = 42

    # Fit the model to the dataset
    losses = rim.fit(
            dataset, 
            learning_rate=learning_rate, 
            checkpoints_directory=checkpoints_directory,
            batch_size=batch_size, 
            checkpoints=1,
            models_to_keep=1,
            epochs=epochs, 
            warmup=warmup, 
            clip=clip, 
            seed=seed
            )
    print(losses)
    # Leave the checkpoints to be removed by last test

    

def test_training_from_checkpoint():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    checkpoints_directory = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"
    # Dont pass model, test that we can load it correctly from checkpoints
    rim = RIM(D, checkpoints_directory=checkpoints_directory, score_fn=score_fn, T=T)
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2*B
        def __getitem__(self, index):
            return torch.randn(C, *D), torch.randn(C, *D)
    
    dataset = Dataset()
    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    batch_size = B
    epochs = 10
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    seed = 42

    # Fit the model to the dataset
    losses = rim.fit(
            dataset, 
            learning_rate=learning_rate, 
            checkpoints_directory=checkpoints_directory,
            models_to_keep=1,
            batch_size=batch_size, 
            epochs=epochs, 
            warmup=warmup, 
            clip=clip, 
            seed=seed
            )
    print(losses)

def test_training_with_ema():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    checkpoints_directory = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"
    rim = RIM(D, checkpoints_directory=checkpoints_directory, score_fn=score_fn, T=T)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2*B
        def __getitem__(self, index):
            return torch.randn(C, *D), torch.randn(C, *D)
    
    dataset = Dataset()

    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    ema_decay=0.99
    batch_size = B
    epochs = 10
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    seed = 42

    # Fit the model to the dataset
    losses = rim.fit(
            dataset, 
            learning_rate=learning_rate, 
            checkpoints_directory=checkpoints_directory,
            models_to_keep=1,
            ema_decay=ema_decay,
            batch_size=batch_size, 
            epochs=epochs, 
            warmup=warmup, 
            clip=clip, 
            seed=seed
            )

    # When test are finished, remove the checkpoint directory
    shutil.rmtree(checkpoints_directory)
