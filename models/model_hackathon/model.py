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


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class Encoder2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Encoder2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, proj_size=output_size)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, f):
        f = f.unsqueeze(1).expand(-1, 24, -1)
        x = torch.cat([x, f], dim=-1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))  # N x L x hidden_size
        return self.fc(out)  # N x L x output_size


class Decoder2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, proj_size=output_size)

    def forward(self, x, hidden):
        out, _ = self.lstm(x, hidden)  # N x L x output_size
        return out


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
    def __init__(self, cp_dim, ep_dim, op_dim, norm_data,
                 num_layers=3, hidden_size=512, forward_version='0'):
        super().__init__()
        self.cp_dim = cp_dim
        self.ep_dim = ep_dim
        self.op_dim = op_dim
        self.norm_data = norm_data
        self.loss_func = nn.MSELoss()

        if forward_version == '0':
            self.forward = self.forward_v2
        elif forward_version == '1':
            self.forward = self.forward_v2_1
        elif forward_version == '2':
            self.forward = self.forward_v2_2
        else:
            raise NotImplementedError(
                f"version {forward_version} not implemented!")

        self.net = build_mlp(
            input_size=self.cp_dim+self.ep_dim+self.op_dim,
            output_size=self.op_dim,
            n_layers=num_layers,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(inplace=True)
        )

    def forward_v2(self, cp, ep, op):
        # all inputs and outputs should be normalized to [0,1]
        x = torch.cat([cp, ep, op], dim=1)
        x = op + self.net(x)
        return torch.clamp(x, 0, 1)

    def forward_v2_1(self, cp, ep, op):
        # all inputs and outputs should be normalized to [0,1]
        x = torch.cat([cp, ep, op], dim=1)
        x = self.net(x)
        return torch.sigmoid(x)

    def forward_v2_2(self, cp, ep, op):
        # all inputs and outputs should be normalized to [0,1]
        x = torch.cat([cp, ep, op], dim=1)
        x = self.net(x)
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
    def __init__(self, op_in_dim, pl_dim, norm_data, num_layers=3, hidden_size=64):
        super().__init__()
        self.op_in_dim = op_in_dim
        self.pl_dim = pl_dim
        self.norm_data = norm_data
        self.loss_func = nn.MSELoss()

        self.net = build_mlp(
            # OP_IN + PL + plant_density -> PL_next
            input_size=self.op_in_dim+self.pl_dim+1,
            output_size=self.pl_dim,
            n_layers=num_layers,
            hidden_size=hidden_size,
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


class ClimateModelDayV3(nn.Module):
    def __init__(self, cp_dim, ep_dim, op_dim, norm_data, version='v3'):
        super(ClimateModelDayV3, self).__init__()
        self.cp_dim = cp_dim // 24
        self.ep_dim = ep_dim // 24
        self.op_dim = op_dim // 24
        self.norm_data = norm_data

        self.encoder_feature_size = 32

        self.encoder = Encoder(
            input_size=self.op_dim,
            hidden_size=32,
            output_size=self.encoder_feature_size,
            num_layers=2
        )

        self.decoder = Decoder(
            input_size=self.cp_dim + self.ep_dim + self.encoder_feature_size,
            hidden_size=64,
            output_size=self.op_dim,
            num_layers=2
        )

    def forward(self, cp, ep, op):
        cp = cp.view(-1, 24, self.cp_dim)  # B x 24 x CP_DIM
        ep = ep.view(-1, 24, self.ep_dim)  # B x 24 x EP_DIM
        op = op.view(-1, 24, self.op_dim)  # B x 24 x OP_DIM

        f = self.encoder(op)  # B x feature_size
        x = torch.cat([cp, ep], dim=-1)  # B x 24 x (CP_DIM + EP_DIM)
        x = self.decoder(x, f)  # B x 24 x OP_DIM
        x = x.view(-1, 24 * self.op_dim)  # B x 24OP_DIM

        return torch.clamp(x, 0, 1)


class ClimateModelDayV4(nn.Module):
    def __init__(self, cp_dim, ep_dim, op_dim, norm_data, version='v4'):
        super(ClimateModelDayV4, self).__init__()
        self.cp_dim = cp_dim // 24
        self.ep_dim = ep_dim // 24
        self.op_dim = op_dim // 24
        self.norm_data = norm_data

        hidden_size = 128

        self.encoder = Encoder2(
            input_size=self.op_dim,
            hidden_size=hidden_size,
            output_size=self.op_dim,
            num_layers=2
        )

        self.decoder = Decoder2(
            input_size=self.cp_dim + self.ep_dim,
            hidden_size=hidden_size,
            output_size=self.op_dim,
            num_layers=2
        )

        if version == 'v4':
            self.forward = self.forward_0
        elif version == 'v4.1':
            self.forward = self.forward_1
        else:
            raise NotImplementedError

    def feature(self, cp, ep, op):
        cp = cp.view(-1, 24, self.cp_dim)  # B x 24 x CP_DIM
        ep = ep.view(-1, 24, self.ep_dim)  # B x 24 x EP_DIM
        op = op.view(-1, 24, self.op_dim)  # B x 24 x OP_DIM

        # num_layers x N x OP_DIM, num_layers x N x hidden_size
        h_n, c_n = self.encoder(op)
        x = torch.cat([cp, ep], dim=-1)  # B x 24 x (CP_DIM + EP_DIM)
        x = self.decoder(x, (h_n, c_n))  # B x 24 x OP_DIM
        return x.flatten(1)  # B x 24OP_DIM

    def forward_0(self, cp, ep, op):
        x = self.feature(cp, ep, op)
        return torch.clamp(x, 0, 1)

    def forward_1(self, cp, ep, op):
        x = self.feature(cp, ep, op)
        return torch.sigmoid(x)

    def predict(self, cp, ep, op):
        # all inputs and outputs are not normalized
        assert cp.shape == (24, self.cp_dim)  # 24 x CP_DIM
        assert ep.shape == (24, self.ep_dim)  # 24 x EP_DIM
        assert op.shape == (24, self.op_dim)  # 24 x OP_DIM

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


MODEL_CONFIGS = {
    'C0': {
        'num_layers': 3,
        'hidden_size': 512,
        'forward_version': '0',
    },
    'C1': {
        'num_layers': 3,
        'hidden_size': 512,
        'forward_version': '1',
    },
    'C1s': {
        'num_layers': 2,
        'hidden_size': 360,
        'forward_version': '1',
    },
    'C2s': {
        'num_layers': 2,
        'hidden_size': 360,
        'forward_version': '2',
    },
    'P0': {
        'num_layers': 3,
        'hidden_size': 64,
    },
}

MODEL_CLASSES = {
    'climate_day': ClimateModelDay,
    'climate_day_v3': ClimateModelDayV3,
    'climate_day_v4': ClimateModelDayV4,
    'plant_day': PlantModelDay,
}


def get_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model_class = MODEL_CLASSES[ckpt['model_class']]
    model_config = ckpt['model_config']
    model = model_class(**model_config)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval()
