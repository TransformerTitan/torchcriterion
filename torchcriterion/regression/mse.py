import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
