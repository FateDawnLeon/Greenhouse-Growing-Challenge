import torch
import torch.nn as nn
import numpy as np

from data import zscore_normalize, zscore_denormalize


def get_spacing_scheme(plant_density):
     # e.g. plant_density = "1 80; 10 40; 20 30; 25 20; 30 10"
    spacing_scheme = []
    day_densities = plant_density.split(';')
    for day_density in day_densities:
        day_density = day_density.strip()
        day, density = day_density.split(' ')
        day, density = int(day), float(density)
        spacing_scheme.append((day, density))
    return spacing_scheme


def get_density(day, spacing_scheme):
    cur = 0
    while cur + 1 < len(spacing_scheme):
        change_day, density = spacing_scheme[cur]
        change_day_next, density_next = spacing_scheme[cur+1]
        if change_day <= day < change_day_next:
            return density
        cur += 1
    return density_next


def get_AvgHeadPerM2(spacing_scheme, num_days):
    return num_days / sum(1/get_density(day, spacing_scheme) for day in range(1, num_days+1))


class Gain:
    @staticmethod
    def compute(FW, DMC, num_days, plant_density):
        spacing_scheme = get_spacing_scheme(plant_density)
        return get_AvgHeadPerM2(spacing_scheme, num_days) * Gain.price(FW, DMC)

    @staticmethod
    def price(FW, DMC):
        price = Gain.std_price(FW)
        if DMC > 0.05:
            price *= 1.1
        elif DMC < 0.045:
            price *= 0.9
        return price

    @staticmethod
    def std_price(FW):
        if FW < 210 or FW > 290:
            return 0
        elif 210 <= FW < 230:
            return 0.4 * (FW - 210) / (230 - 210)
        elif 230 <= FW <= 250:
            return 0.4 + 0.1 * (FW - 230) / (250 - 230)
        else:
            return Gain.std_price(500 - FW)


class FixedCost:
    @staticmethod
    def compute(num_days, pureCO2cap, intensity, num_screens, plant_density):
        spacing_scheme = get_spacing_scheme(plant_density)
        crop_density = get_AvgHeadPerM2(spacing_scheme, num_days)
        num_spacing_changes = len(spacing_scheme) - 1
        fractionOfYear = num_days / 365
        return FixedCost.plant_cost(crop_density) + \
            FixedCost.greenhouse_cost(fractionOfYear) + \
            FixedCost.co2_capacity_cost(fractionOfYear, pureCO2cap) + \
            FixedCost.lamp_cost(fractionOfYear, intensity) + \
            FixedCost.screen_cost(fractionOfYear, num_screens) + \
            FixedCost.spacing_cost(fractionOfYear, num_spacing_changes)

    @staticmethod
    def plant_cost(crop_density):
        return 0.12 * crop_density

    @staticmethod
    def greenhouse_cost(fractionOfYear):
        return 11.5 * fractionOfYear

    @staticmethod
    def co2_capacity_cost(fractionOfYear, pureCO2cap):
        return pureCO2cap * 0.015 * fractionOfYear

    @staticmethod
    def lamp_cost(fractionOfYear, intensity):
        return intensity * 0.0281 * fractionOfYear

    @staticmethod
    def screen_cost(fractionOfYear, num_screens):
        return num_screens * 0.75 * fractionOfYear

    @staticmethod
    def spacing_cost(fractionOfYear, num_spacing_changes):
        return num_spacing_changes * 1.5 * fractionOfYear


class VariableCost:
    @staticmethod
    def compute(comp1_Lmp1_ElecUse, comp1_PConPipe1_Value, comp1_McPureAir_Value, common_Economics_PeakHour):
        return VariableCost.electricity_cost(comp1_Lmp1_ElecUse, common_Economics_PeakHour) + \
            VariableCost.heating_cost(comp1_PConPipe1_Value) + \
            VariableCost.co2_cost(comp1_McPureAir_Value)

    @staticmethod
    def electricity_cost(comp1_Lmp1_ElecUse, common_Economics_PeakHour):
        assert comp1_Lmp1_ElecUse.size() == common_Economics_PeakHour.size()
        OnPeakElec = torch.sum(comp1_Lmp1_ElecUse * common_Economics_PeakHour, dim=-1) / 1000
        OffPeakElec = torch.sum(comp1_Lmp1_ElecUse * (1-common_Economics_PeakHour), dim=-1) / 1000
        return OnPeakElec * 0.1 + OffPeakElec * 0.06

    @staticmethod
    def heating_cost(comp1_PConPipe1_Value):
        return torch.sum(comp1_PConPipe1_Value, dim=-1) / 1000 * 0.03

    @staticmethod
    def co2_cost(comp1_McPureAir_Value):
        kgCO2 = torch.sum(comp1_McPureAir_Value, dim=-1) * 3600
        return kgCO2 * 0.12


def compute_netprofit(control_param, output_param):
    num_days = control_param['num_days']
    plant_density = control_param['plant_density']
    pureCO2cap = control_param['pureCO2cap']
    intensity = control_param['intensity']
    num_screens = control_param['num_screens']
    
    FW = output_param['FW']
    DMC = output_param['DMC']
    comp1_Lmp1_ElecUse = output_param['comp1_Lmp1_ElecUse']
    comp1_PConPipe1_Value = output_param['comp1_PConPipe1_Value']
    comp1_McPureAir_Value = output_param['comp1_McPureAir_Value']
    common_Economics_PeakHour = output_param['common_Economics_PeakHour']
    
    gain = Gain.compute(FW, DMC, num_days, plant_density)
    cost_fixed = FixedCost.compute(num_days, pureCO2cap, intensity, num_screens, plant_density)
    cost_variable = VariableCost.compute(comp1_Lmp1_ElecUse, comp1_PConPipe1_Value, comp1_McPureAir_Value, common_Economics_PeakHour)
    print(gain, cost_fixed, cost_variable)
    return gain - cost_fixed - cost_variable


class Model(nn.Module):

    CP_DIM = 56
    EP_DIM = 5
    OP_NON_PLANT_DIM = 15

    PARAM_BOUNDS = {
        'comp1.Air.T': [-10, 40],
        'comp1.Air.RH': [0, 100],
        'comp1.Air.ppm': [0, 1200],
        'comp1.PARsensor.Above': [0, 1200],
        'comp1.Lmp1.ElecUse': [0, 100],
        'comp1.PConPipe1.Value': [0, float('inf')],
        'comp1.McPureAir.Value': [0, 1e-5],
        'comp1.TPipe1.Value': [0, 100],  # maybe ignoring these params would be better
        'comp1.ConPipes.TSupPipe1': [0, 100],  # maybe ignoring these params would be better
        'comp1.ConWin.WinLee': [0, 100],  # maybe ignoring these params would be better
        'comp1.ConWin.WinWnd': [0, 100],  # maybe ignoring these params would be better
        'comp1.Setpoints.SpHeat': [0, 30],  # maybe ignoring these params would be better
        'comp1.Setpoints.SpVent': [10, 30],  # maybe ignoring these params would be better
        'comp1.Scr1.Pos': [0, 1],  # maybe ignoring these params would be better
        'comp1.Scr2.Pos': [0, 1],  # maybe ignoring these params would be better
    }

    def __init__(self, cp_dim=None, ep_dim=None, op_dim=None, norm_data=None):
        super(Model, self).__init__()

        self.norm_data = norm_data

        lower_bound = np.asarray([bound[0] for bound in self.PARAM_BOUNDS.values()])
        upper_bound = np.asarray([bound[1] for bound in self.PARAM_BOUNDS.values()])

        if self.norm_data:
            min_val = zscore_normalize(lower_bound, norm_data['op_mean'], norm_data['op_std'])
            max_val = zscore_normalize(upper_bound, norm_data['op_mean'], norm_data['op_std'])
            self.min_val = torch.from_numpy(min_val).float()
            self.max_val = torch.from_numpy(max_val).float()

        self.CP_DIM = cp_dim if cp_dim is not None else self.CP_DIM
        self.EP_DIM = ep_dim if ep_dim is not None else self.EP_DIM
        self.OP_NON_PLANT_DIM = op_dim if op_dim is not None else self.OP_NON_PLANT_DIM

        self.in_features = self.CP_DIM + self.EP_DIM + self.OP_NON_PLANT_DIM
        self.out_features = self.OP_NON_PLANT_DIM

        # self.net = nn.Sequential(
        #     nn.Linear(self.in_features, 64),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(64, self.out_features),
        # )
        self.net = nn.Sequential(
            nn.Linear(self.in_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, self.out_features),
        )

    def forward(self, cp, ep_prev, op_prev):
        x = torch.cat([cp, ep_prev, op_prev], dim=1) # B x (cp_dim + ep_dim + op_dim)
        return self.net(x)

    def predict_op(self, cp, ep_prev, op_prev):
        if type(cp) == np.ndarray:
            cp = torch.from_numpy(cp)
        if type(ep_prev) == np.ndarray:
            ep_prev = torch.from_numpy(ep_prev)
        if type(op_prev) == np.ndarray:
            op_prev = torch.from_numpy(op_prev)

        assert type(cp) == torch.Tensor
        assert type(ep_prev) == torch.Tensor
        assert type(op_prev) == torch.Tensor
        
        if len(cp.shape) == 1:
            cp = cp.unsqueeze(0)
        if len(ep_prev.shape) == 1:
            ep_prev = ep_prev.unsqueeze(0)
        if len(op_prev.shape) == 1:
            op_prev = op_prev.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            op_cur = self.forward(cp, ep_prev, op_prev)
            op_cur = torch.maximum(op_cur, self.min_val)
            op_cur = torch.minimum(op_cur, self.max_val)

        return op_cur.cpu().numpy().flatten()

    
    def rollout_multistep(self, cp_cur, ep_pre, op_pre_0):
        self.net.eval()

        T = cp_cur.shape[0]

        # cp_cur: np.ndarray -> T x num_cp_params 
        # ep_pre: np.ndarray -> T x num_ep_params
        # op_pre_1: np.ndarray -> num_op_params

        op = zscore_normalize(op_pre_0, self.norm_data['op_mean'], self.norm_data['op_std'])
        op = torch.from_numpy(op).unsqueeze(0)
        op_cur_prediction = []
        for i in range(T):
            cp = zscore_normalize(cp_cur[i], self.norm_data['cp_mean'], self.norm_data['cp_std'])
            ep = zscore_normalize(ep_pre[i], self.norm_data['ep_mean'], self.norm_data['ep_std'])
            
            cp = torch.from_numpy(cp).unsqueeze(0)
            ep = torch.from_numpy(ep).unsqueeze(0)
            
            with torch.no_grad():
                op = self.forward(cp, ep, op)
                op = torch.maximum(op, self.min_val)
                op = torch.minimum(op, self.max_val)
            op_pred = zscore_denormalize(op.flatten().cpu().numpy(), self.norm_data['op_mean'], self.norm_data['op_std'])
            np.round(op_pred)
            # op[-2:] = (op[-2:]>self.norm_data['op_mean'][-2:]).astype(np.float32)

            op_cur_prediction.append(op_pred)

        return np.asarray(op_cur_prediction, dtype=np.float32) # T x num_op_params
    
    
    def rollout_onestep(self, cp_cur, ep_pre, op_pre):
        self.net.eval()

        T = cp_cur.shape[0]

        # cp_cur: np.ndarray -> T x num_cp_params 
        # ep_pre: np.ndarray -> T x num_ep_params
        # op_pre_1: np.ndarray -> num_op_params

        op_cur_prediction = []
        for i in range(T):
            cp = zscore_normalize(cp_cur[i], self.norm_data['cp_mean'], self.norm_data['cp_std'])
            ep = zscore_normalize(ep_pre[i], self.norm_data['ep_mean'], self.norm_data['ep_std'])
            op = zscore_normalize(op_pre[i], self.norm_data['op_mean'], self.norm_data['op_std'])
            
            cp = torch.from_numpy(cp).unsqueeze(0)
            ep = torch.from_numpy(ep).unsqueeze(0)
            op = torch.from_numpy(op).unsqueeze(0)
            
            with torch.no_grad():
                op = self.forward(cp, ep, op)
                op = torch.maximum(op, self.min_val)
                op = torch.minimum(op, self.max_val)
            op_pred = zscore_denormalize(op.flatten().cpu().numpy(), self.norm_data['op_mean'], self.norm_data['op_std'])
            op_cur_prediction.append(op_pred)

        op_pred_arr = np.asarray(op_cur_prediction, dtype=np.float32) # T x num_op_params

        return np.around(op_pred_arr)


class ModelPlant(nn.Module):

    CP_DIM = 56
    EP_DIM = 5
    OP_OTHER_DIM = 15
    OP_PLANT_DIM = 4

    def __init__(self, cp_dim=None, ep_dim=None, op_other_dim=None, op_plant_dim=None, norm_data=None):
        super(ModelPlant, self).__init__()

        if norm_data:
            min_val = - norm_data['op_plant_mean'] / norm_data['op_plant_std']
            self.min_val = torch.from_numpy(min_val)
        else:
            self.min_val = None

        self.CP_DIM = cp_dim if cp_dim is not None else self.CP_DIM
        self.EP_DIM = ep_dim if ep_dim is not None else self.EP_DIM
        self.OP_OTHER_DIM = op_other_dim if op_other_dim is not None else self.OP_OTHER_DIM
        self.OP_PLANT_DIM = op_plant_dim if op_plant_dim is not None else self.OP_PLANT_DIM
        
        self.in_features = (self.CP_DIM + self.EP_DIM + self.OP_OTHER_DIM) * 24 + self.OP_PLANT_DIM
        self.out_features = self.OP_PLANT_DIM

        self.net = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, self.out_features),
        )

    def forward(self, cp_last_24h, ep_last_24h, op_other_last_24h, op_plant_last_day):
        _, hour_cp, ndim_cp = cp_last_24h.shape
        _, hour_ep, ndim_ep = ep_last_24h.shape
        _, hour_op_other, ndim_op_other = op_other_last_24h.shape
        _, ndim_op_plant = op_plant_last_day.shape

        assert hour_cp == hour_ep == hour_op_other == 24
        assert ndim_cp == self.CP_DIM and ndim_ep == self.EP_DIM and \
            ndim_op_other == self.OP_OTHER_DIM and ndim_op_plant == self.OP_PLANT_DIM

        x = torch.cat([cp_last_24h, ep_last_24h, op_other_last_24h], dim=2)
        x = x.flatten(1)
        x = torch.cat([x, op_plant_last_day], dim=1)

        assert x.shape[1] == self.in_features

        x = self.net(x)

        if self.min_val is not None:
            x = torch.maximum(x, self.min_val)

        return x

    def predict_op(self, cp_last_24h, ep_last_24h, op_other_last_24h, op_plant_last_day):
        if isinstance(cp_last_24h, np.ndarray):
            cp_last_24h = torch.from_numpy(cp_last_24h)
        if isinstance(ep_last_24h, np.ndarray):
            ep_last_24h = torch.from_numpy(ep_last_24h)
        if isinstance(op_other_last_24h, np.ndarray):
            op_other_last_24h = torch.from_numpy(op_other_last_24h)
        if isinstance(op_plant_last_day, np.ndarray):
            op_plant_last_day = torch.from_numpy(op_plant_last_day)

        assert type(cp_last_24h) == torch.Tensor
        assert type(ep_last_24h) == torch.Tensor
        assert type(op_other_last_24h) == torch.Tensor
        assert type(op_plant_last_day) == torch.Tensor

        assert len(cp_last_24h.shape) == 2
        assert cp_last_24h.shape[0] == 24
        assert cp_last_24h.shape[1] == self.CP_DIM
        assert len(ep_last_24h.shape) == 2
        assert ep_last_24h.shape[0] == 24
        assert ep_last_24h.shape[1] == self.EP_DIM
        assert len(op_other_last_24h.shape) == 2
        assert op_other_last_24h.shape[0] == 24
        assert op_other_last_24h.shape[1] == self.OP_OTHER_DIM
        assert len(op_plant_last_day.shape) == 1
        assert op_plant_last_day.shape[0] == self.OP_PLANT_DIM

        cp_last_24h = cp_last_24h.unsqueeze(0)
        ep_last_24h = ep_last_24h.unsqueeze(0)
        op_other_last_24h = op_other_last_24h.unsqueeze(0)
        op_plant_last_day = op_plant_last_day.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            op_cur = self.forward(cp_last_24h, ep_last_24h, op_other_last_24h, op_plant_last_day)
            op_cur = op_cur.detach().cpu().numpy().flatten()

        return op_cur

    def rollout(self, cp, ep, op_other, op_plant_1):
        self.net.eval()

        # cp, ep, op_other -> T_days x 24 x ndim_{cp,ep,op_other}
        # op_plant_1 -> ndim_op_plant_1

        for arr in [cp, ep, op_other, op_plant_1]:
            assert isinstance(arr, np.ndarray)

        cp = torch.from_numpy(cp).float()
        ep = torch.from_numpy(ep).float()
        op_other = torch.from_numpy(op_other).float()
        op_plant_1 = torch.from_numpy(op_plant_1).float()
        
        days_cp, hour_cp, ndim_cp = cp.shape
        days_ep, hour_ep, ndim_ep = ep.shape
        days_op_other, hour_op_other, ndim_op_other = op_other.shape
        ndim_op_plant = len(op_plant_1)

        assert days_cp == days_ep == days_op_other
        assert hour_cp == hour_ep == hour_op_other
        assert ndim_cp == self.CP_DIM and ndim_ep == self.EP_DIM and \
            ndim_op_other == self.OP_OTHER_DIM and ndim_op_plant == self.OP_PLANT_DIM

        op_plant_all = []
        op_plant_last_day = op_plant_1.unsqueeze(0)
        for i in range(days_cp):
            cp_last_24h = cp[i:i+1]
            ep_last_24h = ep[i:i+1]
            op_other_last_24h = op_other[i:i+1]
            with torch.no_grad():        
                op_plant_last_day = self.forward(cp_last_24h, ep_last_24h, op_other_last_24h, op_plant_last_day)
            op_plant_all.append(op_plant_last_day)

        return torch.cat(op_plant_all, dim=0).cpu().numpy()



class SuperGreenhouseModel(nn.Module):
    
    CP_DIM = None
    EP_DIM = 4 * 24
    OP_PL_DIM = 4
    OP_IN_DIM = 7 * 24
    OP_DIM = OP_IN_DIM + OP_PL_DIM

    EP = [
        'common.Iglob.Value',
        'common.TOut.Value',
        'common.RHOut.Value',
        'common.Windsp.Value',
    ]
    OP_IN = [
        "comp1.Air.T",
        "comp1.Air.RH",
        "comp1.Air.ppm",
        "comp1.PARsensor.Above",
    ]
    OP_PL = [
        "comp1.Plant.headFW",
        "comp1.Plant.shootDryMatterContent",
        "comp1.Plant.fractionGroundCover",
        "comp1.Plant.plantProjection",
    ]
    OP_COST = [
        "comp1.Lmp1.ElecUse",
        "comp1.PConPipe1.Value",
        "comp1.McPureAir.Value",
    ]
    
    def __init__(self):
        super(SuperGreenhouseModel, self).__init__()