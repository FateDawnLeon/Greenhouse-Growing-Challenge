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
            op_in_next = self.forward(
                cp_tensor, ep_tensor, op_in_tensor).squeeze().numpy()

        return unnormalize(op_in_next, op_in_mean, op_in_std)


class PlantModel(nn.Module):
    def __init__(self, op_in_dim, pl_dim):
        super().__init__()
        self.op_in_dim = op_in_dim
        self.pl_dim = pl_dim

        class OutputActivation(nn.Module):
            def __init__(self):
                super().__init__()
                self.threshold = torch.tensor([0, -1, 0])

            def forward(self, x):
                return torch.max(x, self.threshold)

        self.net = build_mlp(
            input_size=self.op_in_dim + self.pl_dim,
            output_size=self.pl_dim,
            n_layers=3,
            hidden_size=64,
            activation=nn.LeakyReLU(inplace=True),
            output_activation=OutputActivation()
        )

    def forward(self, op_in, pl):  # all inputs and output are normalized
        x = torch.cat([op_in, pl], dim=1)
        return pl + self.net(x)

    def predict(self, op_in, pl, norm_data):  # all inputs and output are unnormalized
        op_in_mean, op_in_std = norm_data['op_in_mean'], norm_data['op_in_std']
        op_pl_mean, op_pl_std = norm_data['op_pl_mean'], norm_data['op_pl_std']

        op_in_normed = normalize(op_in, op_in_mean, op_in_std)
        op_pl_normed = normalize(pl, op_pl_mean, op_pl_std)

        with torch.no_grad():
            op_in_tensor = torch.from_numpy(op_in_normed).float().unsqueeze(0)
            op_pl_tensor = torch.from_numpy(op_pl_normed).float().unsqueeze(0)
            op_pl_next = self.forward(
                op_in_tensor, op_pl_tensor).squeeze().numpy()

        return unnormalize(op_pl_next, op_pl_mean, op_pl_std)


class ClimateModelDay(nn.Module):
    def __init__(self, cp_dim, ep_dim, op_dim, norm_data):
        super().__init__()
        self.cp_dim = cp_dim
        self.ep_dim = ep_dim
        self.op_dim = op_dim
        self.norm_data = norm_data
        self.loss_func = nn.MSELoss()

        self.net = build_mlp(
            input_size=self.cp_dim+self.ep_dim+self.op_dim,
            output_size=self.op_dim,
            n_layers=3,
            hidden_size=512,
            activation=nn.LeakyReLU(inplace=True)
        )

    def forward(self, cp, ep, op):
        # all inputs and outputs should be normalized to [0,1]
        x = torch.cat([cp, ep, op], dim=1)
        x = op + self.net(x)
        return torch.clamp(x, 0, 1)

    def predict(self, cp, ep, op):
        # all inputs and outputs are not normalized
        assert cp.shape == (24, self.cp_dim // 24)  # 24 x CP_DIM
        assert ep.shape == (24, self.ep_dim // 24)  # 24 x EP_DIM
        assert op.shape == (24, self.op_dim // 24)  # 24 x OP_DIM

        cp = normalize_zero2one(cp, self.norm_data['cp']).flatten()
        ep = normalize_zero2one(ep, self.norm_data['ep']).flatten()
        op = normalize_zero2one(op, self.norm_data['op']).flatten()

        with torch.no_grad():
            cp = make_tensor(cp)
            ep = make_tensor(ep)
            op = make_tensor(op)
            op_next = self.forward(cp, ep, op).squeeze().numpy()
            op_next = op_next.reshape(24, -1)  # 24 x OP_DIM

        # 24 x OP_DIM
        return unnormalize_zero2one(op_next, self.norm_data['op'])


class PlantModelDay(nn.Module):
    def __init__(self, op_in_dim, pl_dim, norm_data):
        super().__init__()
        self.op_in_dim = op_in_dim
        self.pl_dim = pl_dim
        self.norm_data = norm_data
        self.loss_func = nn.MSELoss()

        self.net = build_mlp(
            # OP_IN + PL + plant_density -> PL_next
            input_size=self.op_in_dim+self.pl_dim+1,
            output_size=self.pl_dim,
            n_layers=3,
            hidden_size=64,
            activation=nn.LeakyReLU(inplace=True),
        )

    def forward(self, pd, op_in, pl):
        # all inputs and outputs should be normalized to [0,1]
        x = torch.cat([pd, op_in, pl], dim=1)
        x = pl + self.net(x)
        return torch.clamp(x, 0, 1)

    def predict(self, pd, op_in, pl):
        # all inputs and outputs are not normalized
        assert op_in.shape == (24, self.op_in_dim // 24)  # 24 x OP_IN_DIM
        assert pd.shape == (1,)  # 1
        assert pl.shape == (self.pl_dim,)  # PL_DIM

        op_in = normalize_zero2one(op_in, self.norm_data['op_in']).flatten()
        pd = normalize_zero2one(pd, self.norm_data['pd'])
        pl = normalize_zero2one(pl, self.norm_data['pl'])

        with torch.no_grad():
            op_in = make_tensor(op_in)
            pd = make_tensor(pd)
            pl = make_tensor(pl)
            pl_next = self.forward(pd, op_in, pl).squeeze().numpy()  # PL_DIM

        return unnormalize_zero2one(pl_next, self.norm_data['pl'])  # PL_DIM
