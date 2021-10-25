import torch
import torch.nn as nn
from utils import normalize, unnormalize, normalize_zero2one, unnormalize_zero2one, make_tensor


def build_mlp(
    input_size,
    output_size,
    n_layers,
    hidden_size,
    activation,
    output_activation=None
):
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(activation)
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ClimateModel(nn.Module):
    def __init__(self, cp_dim, ep_dim, op_in_dim, criterion=nn.MSELoss()):
        super().__init__()
        self.cp_dim = cp_dim
        self.ep_dim = ep_dim
        self.op_in_dim = op_in_dim
        self.criterion = criterion
        self.net = build_mlp(
            input_size=self.cp_dim + self.ep_dim + self.op_in_dim,
            output_size=self.op_in_dim, 
            n_layers=3, 
            hidden_size=64,
            activation=nn.LeakyReLU(inplace=True)
        )

    def forward(self, cp, ep, op_in):  # all inputs and output are normalized
        x = torch.cat([cp, ep, op_in], dim=1)
        return op_in + self.net(x)

    def predict(self, cp, ep, op_in, norm_data):  # all inputs and output are unnormalized
        cp_mean, cp_std = norm_data['cp_mean'], norm_data['cp_std']
        ep_mean, ep_std = norm_data['ep_mean'], norm_data['ep_std']
        op_in_mean, op_in_std = norm_data['op_in_mean'], norm_data['op_in_std']

        cp_normed = normalize(cp, cp_mean, cp_std)
        ep_normed = normalize(ep, ep_mean, ep_std)
        op_in_normed = normalize(op_in, op_in_mean, op_in_std)

        with torch.no_grad():
            cp_tensor = torch.from_numpy(cp_normed).float().unsqueeze(0)
            ep_tensor = torch.from_numpy(ep_normed).float().unsqueeze(0)
            op_in_tensor = torch.from_numpy(op_in_normed).float().unsqueeze(0)
            op_in_next = self.forward(cp_tensor, ep_tensor, op_in_tensor).squeeze().numpy()
    
        return unnormalize(op_in_next, op_in_mean, op_in_std)


class PlantModel(nn.Module):
    def __init__(self, op_in_dim, op_pl_dim):
        super().__init__()
        self.op_in_dim = op_in_dim
        self.op_pl_dim = op_pl_dim
        
        class OutputActivation(nn.Module):
            def __init__(self):
                super().__init__()
                self.threshold = torch.tensor([0, -1, 0])

            def forward(self, x):
                return torch.max(x, self.threshold)
        
        self.net = build_mlp(
            input_size=self.op_in_dim + self.op_pl_dim,
            output_size=self.op_pl_dim, 
            n_layers=3, 
            hidden_size=64,
            activation=nn.LeakyReLU(inplace=True),
            output_activation=OutputActivation()
        )

    def forward(self, op_in, op_pl):  # all inputs and output are normalized
        x = torch.cat([op_in, op_pl], dim=1)
        return op_pl + self.net(x)

    def predict(self, op_in, op_pl, norm_data):  # all inputs and output are unnormalized
        op_in_mean, op_in_std = norm_data['op_in_mean'], norm_data['op_in_std']
        op_pl_mean, op_pl_std = norm_data['op_pl_mean'], norm_data['op_pl_std']

        op_in_normed = normalize(op_in, op_in_mean, op_in_std)
        op_pl_normed = normalize(op_pl, op_pl_mean, op_pl_std)

        with torch.no_grad():
            op_in_tensor = torch.from_numpy(op_in_normed).float().unsqueeze(0)
            op_pl_tensor = torch.from_numpy(op_pl_normed).float().unsqueeze(0)
            op_pl_next = self.forward(op_in_tensor, op_pl_tensor).squeeze().numpy()

        return unnormalize(op_pl_next, op_pl_mean, op_pl_std)


class ClimateModelDay(nn.Module):
    def __init__(self, cp_dim, ep_dim, op_dim, ranges):
        super().__init__()
        self.cp_dim = cp_dim
        self.ep_dim = ep_dim
        self.op_dim = op_dim
        self.ranges = ranges
        
        self.loss_func = nn.MSELoss()

        self.net = build_mlp(
            input_size=self.cp_dim+self.ep_dim+self.op_dim,
            output_size=self.op_dim,
            n_layers=3,
            hidden_size=128,
            activation=nn.LeakyReLU(inplace=True)
        )

    def forward(self, cp, ep, op):  # all inputs should be normalized to [0,1]
        x = torch.cat([cp, ep, op], dim=1)
        x = op + self.net(x)
        return torch.clamp(x, 0, 1)  # all outputs are clamped to lie in [0,1]

    def predict(self, cp, ep, op):  # all inputs are not normalized
        cp = normalize_zero2one(cp, self.ranges['cp']).flatten()  # 24 x CP_DIM
        ep = normalize_zero2one(ep, self.ranges['ep']).flatten()  # 24 x EP_DIM
        op = normalize_zero2one(op, self.ranges['op']).flatten()  # 24 x OP_DIM

        with torch.no_grad():
            cp = make_tensor(cp)
            ep = make_tensor(ep)
            op = make_tensor(op)
            op_next = self.forward(cp, ep, op).squeeze().numpy()

        return unnormalize_zero2one(op_next.reshape(24, -1), self.ranges['op'])  # all outputs are unnormalized


class PlantModelDay(nn.Module):
    def __init__(self, op_in_dim, op_pl_dim, ranges):
        super().__init__()
        self.op_in_dim = op_in_dim
        self.op_pl_dim = op_pl_dim
        self.ranges = ranges
        self.loss_func = nn.MSELoss()

        self.net = build_mlp(
            input_size=self.op_in_dim+self.op_pl_dim,
            output_size=self.op_pl_dim,
            n_layers=3,
            hidden_size=128,
            activation=nn.LeakyReLU(inplace=True),
        )

    def forward(self, op_in, op_pl):  # all inputs should be normalized to [0,1]
        x = torch.cat([op_in, op_pl], dim=1)
        x = op_pl + self.net(x)
        return torch.clamp(x, 0, 1)  # all outputs are clamped to lie in [0,1]

    def predict(self, op_in, op_pl):  # all inputs are not normalized
        op_in = normalize_zero2one(op_in, self.ranges['op_in']).flatten()  # 24 x EP_DIM
        op_pl = normalize_zero2one(op_pl, self.ranges['op_pl']).flatten()  # 24 x CP_DIM

        with torch.no_grad():
            op_in = make_tensor(op_in)
            op_pl = make_tensor(op_pl)
            op_pl_next = self.forward(op_in, op_pl).squeeze().numpy()

        return unnormalize_zero2one(op_pl_next, self.ranges['op_pl'])  # all outputs are unnormalized
