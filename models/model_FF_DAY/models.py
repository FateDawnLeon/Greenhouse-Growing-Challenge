import torch
import torch.nn as nn


class ModelClimate(nn.Module):

    EP_OUT_KEYS = [
        'common.Iglob.Value',
        'common.TOut.Value',
        'common.RHOut.Value',
        'common.Windsp.Value',
    ]
    EP_IN_WEATHER_KEYS = [
        "comp1.Air.T",
        "comp1.Air.RH",
        "comp1.Air.ppm",
        "comp1.PARsensor.Above",
    ]
    EP_IN_OTHER_KEYS = [
        "comp1.TPipe1.Value",
        "comp1.ConPipes.TSupPipe1",
        "comp1.PConPipe1.Value",
        "comp1.ConWin.WinLee",
        "comp1.ConWin.WinWnd",
        "comp1.Setpoints.SpHeat",
        "comp1.Setpoints.SpVent",
        "comp1.Scr1.Pos",
        "comp1.Scr2.Pos",
        "comp1.Lmp1.ElecUse",
        "comp1.McPureAir.Value",
    ]

    def __init__(self, cp_dim, hidden_dim=128):
        super().__init__()
        self.ep_out_dim = len(self.EP_OUT_KEYS)
        self.ep_in_weather_dim = len(self.EP_IN_WEATHER_KEYS)
        self.ep_in_other_dim = len(self.EP_IN_OTHER_KEYS)
        self.cp_dim = cp_dim

        input_dim = self.ep_out_dim + self.ep_in_weather_dim + cp_dim
        output_dim = self.ep_in_weather_dim
        self.net_weather = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        input_dim = self.ep_out_dim + self.ep_in_other_dim + cp_dim
        output_dim = self.ep_in_other_dim
        self.net_other = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch):
        """
        Input:
            ep_out: outside weather params
            ep_in: inside weather params
            cp: control parmas
        Output:
            ep_in_next: inside weather params of next timestep
        """
        ep_out = batch['ep_out']
        ep_in_weather = batch['ep_in_weather']
        ep_in_other = batch['ep_in_other']
        cp = batch['cp']

        input = torch.cat([ep_out, ep_in_weather, cp], dim=1)
        ep_in_weather_next = ep_in_weather + self.net_weather(input)

        input = torch.cat([ep_out, ep_in_other, cp], dim=1)
        ep_in_other_next = ep_in_other + self.net_other(input)

        return ep_in_weather_next, ep_in_other_next


class ModelPlant(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, wp_in, op_pl):
        """
        Input:
            wp_in: inside weather params
            op_pl: plant params
        Output:
            op_pl_next: plant params of next timestep
        """


if __name__ == "__main__":
    batch_size = 64
    model = ModelClimate(cp_dim=50)
    batch = dict(
        ep_out=torch.rand(batch_size, model.ep_out_dim),
        ep_in_weather=torch.rand(batch_size, model.ep_in_weather_dim),
        ep_in_other=torch.rand(batch_size, model.ep_in_other_dim),
        cp=torch.rand(batch_size, model.cp_dim),
    )

    ep_in_weather_next, ep_in_other_next = model(batch)
    print(ep_in_weather_next.shape)
    print(ep_in_other_next.shape)
