import torch
from rim import RIM, Hourglass

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
    rim = RIM(model, D, score_fn, T=T)
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
    epsilon = 0 # avoid t=0 in sde sample (not needed for VESDE)
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    seed = 42

    # Fit the model to the dataset
    losses = rim.fit(
            dataset, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            epochs=epochs, 
            epsilon=epsilon, 
            warmup=warmup, 
            clip=clip, 
            seed=seed
            )
    print(losses)
    assert 0 == 1
