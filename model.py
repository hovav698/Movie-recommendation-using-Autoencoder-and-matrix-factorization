import torch

class AutoEncoder(torch.nn.Module):
    def __init__(self, M, K):
        super(AutoEncoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(M, K),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(K, M),
            torch.nn.ReLU()
        )

    def forward(self, input):
        return self.model(input)
