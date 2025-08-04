import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hiddens, n_layers):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(n_hiddens, n_hiddens), nn.ReLU()) for _ in range(n_layers - 1)],
            nn.Linear(n_hiddens, n_outputs)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
