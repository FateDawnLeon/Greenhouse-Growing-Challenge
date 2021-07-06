import torch
from torch import nn
import numpy as np


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
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, out_features),
        )

    def forward(self, cp, ep, op_pre):
        x = torch.cat([cp, ep, op_pre], dim=1) # B x (cp_dim + ep_dim + op_dim)
        return self.net(x)
    
    # def forward(self, cp, ep, op_pre):
    #     op_dim = op_pre.shape[1]

    #     x = torch.cat([cp, ep, op_pre], dim=1) # B x (cp_dim + ep_dim + op_dim)
    #     diff = self.net(x)

    #     incre_op_idx = [4, 19]

    #     positive_diff = diff[:, incre_op_idx]
    #     other_diff = diff[:, [i for i in range(op_dim) if i not in incre_op_idx]]
    #     diff = torch.cat([torch.relu(positive_diff), other_diff])
        
    #     return diff + op_pre

    def predict_op(self, cp, ep, op_pre):
        if type(cp) == np.ndarray:
            cp = torch.from_numpy(cp)
        if type(ep) == np.ndarray:
            ep = torch.from_numpy(ep)
        if type(op_pre) == np.ndarray:
            op_pre = torch.from_numpy(op_pre)

        assert type(cp) == torch.Tensor
        assert type(ep) == torch.Tensor
        assert type(op_pre) == torch.Tensor
        
        if len(cp.shape) == 1:
            cp = cp.unsqueeze(0)
        if len(ep.shape) == 1:
            ep = ep.unsqueeze(0)
        if len(op_pre.shape) == 1:
            op_pre = op_pre.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            op_cur = self.forward(cp, ep, op_pre).detach().cpu().numpy()

        return op_cur
    
    def inference_output_episode(self, cp, ep, op_0):
        self.eval()

        # cp: np.ndarray -> T x num_cp_params 
        # ep: np.ndarray -> T x num_ep_params
        # op_0: np.ndarray -> num_op_params

        op_i = torch.from_numpy(op_0).unsqueeze(0)
        cp = torch.from_numpy(cp)
        ep = torch.from_numpy(ep)

        with torch.no_grad():
            op_episode = []
            for cp_i, ep_i in zip(cp, ep):
                cp_i = cp_i.unsqueeze(0)    
                ep_i = ep_i.unsqueeze(0)    
                op_i = self(cp_i, ep_i, op_i)
                op_episode.append(op_i)

        return torch.cat(op_episode, dim=0).cpu().numpy() # T x num_op_params


class ModelPlant(nn.Module):
    def __init__(self):
        super(ModelPlant, self).__init__()

    def forward(self, x):
        pass