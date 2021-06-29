import torch
from torch import nn


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
            price = 0
        elif 210 <= FW < 230:
            price = 0.4 * (FW - 210) / (230 - 210)
        elif 230 <= FW <= 250:
            price = 0.4 + 0.1 * (FW - 230) / (250 - 230)
        else:
            price = Gain.std_price(500 - FW)
        return price


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

    CONTROL_KEYS = [
        "simset.@endDate",
        "comp1.heatingpipes.pipe1.@maxTemp",
        "comp1.heatingpipes.pipe1.@minTemp",
        "comp1.heatingpipes.pipe1.@radiationInfluence",
        "comp1.setpoints.temp.@heatingTemp",
        "comp1.setpoints.temp.@ventOffset",
        "comp1.setpoints.temp.@radiationInfluence",
        "comp1.setpoints.temp.@PbandVent",
        "comp1.setpoints.ventilation.@startWnd",
        "comp1.setpoints.ventilation.@winLeeMin",
        "comp1.setpoints.ventilation.@winLeeMax",
        "comp1.setpoints.ventilation.@winWndMin",
        "comp1.setpoints.ventilation.@winWndMax",
        "common.CO2dosing.@pureCO2cap",
        "comp1.setpoints.CO2.@setpoint",
        "comp1.setpoints.CO2.@setpIfLamps",
        "comp1.setpoints.CO2.@doseCapacity",
        "comp1.screens.scr1.@enabled",
        "comp1.screens.scr1.@material",
        "comp1.screens.scr1.@ToutMax",
        "comp1.screens.scr1.@closeBelow",
        "comp1.screens.scr1.@closeAbove",
        "comp1.screens.scr1.@lightPollutionPrevention",
        "comp1.screens.scr2.@enabled",
        "comp1.screens.scr2.@material",
        "comp1.screens.scr2.@ToutMax",
        "comp1.screens.scr2.@closeBelow",
        "comp1.screens.scr2.@closeAbove",
        "comp1.screens.scr2.@lightPollutionPrevention",
        "comp1.illumination.lmp1.@enabled",
        "comp1.illumination.lmp1.@intensity",
        "comp1.illumination.lmp1.@hoursLight",
        "comp1.illumination.lmp1.@endTime",
        "comp1.illumination.lmp1.@maxIglob",
        "comp1.illumination.lmp1.@maxPARsum",
        "crp_lettuce.Intkam.management.@plantDensity",
    ]

    ENV_KEYS = [
        'common.Iglob.Value',
        'common.TOut.Value',
        'common.RHOut.Value',
        'common.Windsp.Value',
        'common.Economics.PeakHour',
    ]

    OUTPUT_KEYS = [
        'comp1.Air.T',
        'comp1.Air.RH',
        'comp1.Air.ppm',
        'comp1.PARsensor.Above',
        'comp1.TPipe1.Value',
        'comp1.ConPipes.TSupPipe1',
        'comp1.PConPipe1.Value',
        'comp1.ConWin.WinLee',
        'comp1.ConWin.WinWnd',
        'comp1.Setpoints.SpHeat',
        'comp1.Setpoints.SpVent',
        'comp1.Scr1.Pos',
        'comp1.Scr2.Pos',
        'comp1.McPureAir.Value',
        'comp1.Plant.headFW',
        'comp1.Plant.fractionGroundCover',
        'comp1.Plant.shootDryMatterContent',
        'comp1.Plant.plantProjection',
        'comp1.Plant.PlantDensity',
        'comp1.Lmp1.ElecUse',
    ]

    def __init__(self):
        super(Model, self).__init__()

        num_in_features = len(Model.OUTPUT_KEYS) + len(Model.CONTROL_KEYS) + len(Model.ENV_KEYS)
        num_out_features = len(Model.OUTPUT_KEYS)
        
        self.net = nn.Sequential(
            nn.Linear(num_in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_out_features),
        )

    def forward(self, cp, w, op):
        x = torch.cat([cp, w, op], dim=-1)
        return self.net(x)