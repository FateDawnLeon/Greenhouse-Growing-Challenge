import torch
import datetime
import numpy as np

from collections import OrderedDict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CITY_NAME = 'Amsterdam'
START_DATE = datetime.date(2021, 3, 4)
MATERIALS = ['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']

# ====================== control param related ======================
CONTROL_RL = OrderedDict([
    # temp_night[10,15], temp_day[15,30]
    ("comp1.setpoints.temp.@heatingTemp", 2),
    ("comp1.setpoints.temp.@ventOffset", 1),  # [0, 5]
    ("comp1.setpoints.ventilation.@startWnd", 1),  # [0, 50]
    ("comp1.setpoints.CO2.@setpoint", 2),
    ("comp1.screens.scr1.@ToutMax", 1),
    ("comp1.screens.scr1.@closeBelow", 1),
    ("comp1.screens.scr1.@closeAbove", 1),
    ("comp1.screens.scr2.@ToutMax", 1),
    ("comp1.screens.scr2.@closeBelow", 1),
    ("comp1.screens.scr2.@closeAbove", 1),
    ("comp1.illumination.lmp1.@endTime", 1),  # [18, 20]
    ("comp1.illumination.lmp1.@hoursLight", 1),  # [0, 18]
    ("crp_lettuce.Intkam.management.@plantDensity", 2)  # (value, change)
])

CONTROL_BO = [
    "simset.@startDate",  # e.g. '2021-03-05'
    "common.CO2dosing.@pureCO2cap",  # e.g. 280
    "comp1.screens.scr1.@enabled",  # e.g. True
    "comp1.screens.scr1.@material",  # e.g. 'scr_Blackout.par'
    "comp1.screens.scr2.@enabled",  # e.g. False
    "comp1.screens.scr2.@material",  # e.g. 'scr_Transparent.par'
    "comp1.illumination.lmp1.@intensity",  # e.g. 100
    "comp1.illumination.lmp1.@maxIglob",  # e.g. 500
]

CONTROL_FIX = OrderedDict([
    ("comp1.heatingpipes.pipe1.@maxTemp", 60),
    ("comp1.heatingpipes.pipe1.@minTemp", 0),
    ("comp1.heatingpipes.pipe1.@radiationInfluence", "100 300"),
    ("comp1.setpoints.temp.@radiationInfluence", "50 150 1"),
    ("comp1.setpoints.temp.@PbandVent", "0 10; 20 5"),
    ("comp1.setpoints.ventilation.@winLeeMin", 0),
    ("comp1.setpoints.ventilation.@winLeeMax", 100),
    ("comp1.setpoints.ventilation.@winWndMin", 0),
    ("comp1.setpoints.ventilation.@winWndMax", 100),
    ("comp1.setpoints.CO2.@setpIfLamps", 0),
    ("comp1.setpoints.CO2.@doseCapacity", "20 100; 40 50; 70 25"),
    ("comp1.illumination.lmp1.@maxPARsum", 50)
])

# ====================== simulator related ======================
KEYS = {
    'A': 'C48A-ZRJQ-3wcq-rGuC-mEme',
    'B': 'C48B-PTmQ-89Kx-jqV5-3zRL',
    'C': 'C48A-ZRJQ-3wcq-rGuC-mEme',
    'D': 'C48B-PTmQ-89Kx-jqV5-3zRL',
    'hack': 'H17-KyEO-iDtD-mVGR',
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/'
SAMPLE_CONTROL_JSON_PATH = './ClimateControlSample.json'

# ====================== data related ======================
TRACES_DIR = ''  # TODO
EP_PATH = ''  # TODO
CLIMATE_MODEL_PATH = ''  # TODO
PLANT_MODEL_PATH = ''  # TODO
BO_CONTROL_PATH = ''  # TODO

# ====================== runtime related ======================
ACTION_RARAM_SPACE = {
    "end": ([0], [1]),
    "comp1.setpoints.temp.@heatingTemp": ([0, 15], [15, 30]),  # [N_min, D_min], [N_max, D_max]
    "comp1.setpoints.temp.@ventOffset": ([0], [5]),
    "comp1.setpoints.ventilation.@startWnd": ([0], [100]),
    "comp1.setpoints.CO2.@setpoint": ([0, 400], [400, 1200]),  # [N_min, D_min], [N_max, D_max]
    "comp1.screens.scr1.@ToutMax": ([-20], [30]),
    "comp1.screens.scr1.@closeBelow": ([0], [200]),
    "comp1.screens.scr1.@closeAbove": ([500], [1500]),
    "comp1.screens.scr2.@ToutMax": ([-20], [30]),
    "comp1.screens.scr2.@closeBelow": ([0], [200]),
    "comp1.screens.scr2.@closeAbove": ([500], [1500]),
    "comp1.illumination.lmp1.@hoursLight": ([0], [18]),
    "comp1.illumination.lmp1.@endTime": ([18], [20]),
    "crp_lettuce.Intkam.management.@plantDensity": ([0, 0], [1, 35]),  # [change_min, value_min], [change_max, value_max]
}

MODEL_PARAM_SPACE = {
    # CP params -> RL
    "comp1.setpoints.temp.@heatingTemp": ([0, 0, 0, 0, 0, 0, 0, 0], [24, 30, 24, 30, 24, 30, 24, 30]),
    "comp1.setpoints.temp.@ventOffset": ([0], [5]),
    "comp1.setpoints.ventilation.@startWnd": ([0], [100]),
    "comp1.setpoints.CO2.@setpoint": ([0, 0, 0, 0, 0, 0, 0, 0], [24, 1200, 24, 1200, 24, 1200, 24, 1200]),
    "comp1.screens.scr1.@ToutMax": ([-20], [30]),
    "comp1.screens.scr1.@closeBelow": ([0], [200]),
    "comp1.screens.scr1.@closeAbove": ([500], [1500]),
    "comp1.screens.scr2.@ToutMax": ([-20], [30]),
    "comp1.screens.scr2.@closeBelow": ([0], [200]),
    "comp1.screens.scr2.@closeAbove": ([500], [1500]),
    "comp1.illumination.lmp1.@hoursLight": ([0], [18]),
    "comp1.illumination.lmp1.@endTime": ([18], [20]),
    "crp_lettuce.Intkam.management.@plantDensity": ([0], [100]),
    # CP params -> BO
    "common.CO2dosing.@pureCO2cap": ([100], [500]),
    "comp1.screens.scr1.@enabled": ([0, 0], [1, 1]),
    "comp1.screens.scr1.@material": ([0, 0, 0], [1, 1, 1]),
    "comp1.screens.scr2.@enabled": ([0, 0], [1, 1]),
    "comp1.screens.scr2.@material": ([0, 0, 0], [1, 1, 1]),
    "comp1.illumination.lmp1.@intensity": ([0], [500]),
    "comp1.illumination.lmp1.@maxIglob": ([0], [500]),
    # EP params
    'common.Iglob.Value': ([0], [1000]),
    'common.TOut.Value': ([-10], [40]),
    'common.RHOut.Value': ([0], [100]),
    'common.Windsp.Value': ([0], [10]),
    # OP params
    'comp1.Air.T': ([0], [30]),
    'comp1.Air.RH': ([0], [100]),
    'comp1.Air.ppm': ([0], [1200]),
    'comp1.PARsensor.Above': ([0], [2000]),
    'comp1.Plant.PlantDensity': ([0], [100]),
    "comp1.Plant.fractionGroundCover": ([0], [1]),
    'comp1.TPipe1.Value': ([0], [60]),
    'comp1.ConPipes.TSupPipe1': ([0], [60]),
    'comp1.PConPipe1.Value': ([0], [200]),
    'comp1.ConWin.WinLee': ([0], [100]),
    'comp1.ConWin.WinWnd': ([0], [100]),
    'comp1.Setpoints.SpHeat': ([0], [40]),
    'comp1.Setpoints.SpVent': ([0], [30]),
    'comp1.Scr1.Pos': ([0], [1]),
    'comp1.Scr2.Pos': ([0], [1]),
    'comp1.Lmp1.ElecUse': ([0], [70]),
    'comp1.McPureAir.Value': ([0], [1e-5]),
    # PL params
    'comp1.Plant.headFW': ([0], [1000]),
    'comp1.Plant.shootDryMatterContent': ([0], [0.1]),
    'comp1.Plant.qualityLoss': ([0], [100]),
}

CP_KEYS = [
    # RL params
    "comp1.setpoints.temp.@heatingTemp",
    "comp1.setpoints.temp.@ventOffset",
    "comp1.setpoints.ventilation.@startWnd",
    "comp1.setpoints.CO2.@setpoint",
    "comp1.screens.scr1.@ToutMax",
    "comp1.screens.scr1.@closeBelow",
    "comp1.screens.scr1.@closeAbove",
    "comp1.screens.scr2.@ToutMax",
    "comp1.screens.scr2.@closeBelow",
    "comp1.screens.scr2.@closeAbove",
    "comp1.illumination.lmp1.@hoursLight",
    "comp1.illumination.lmp1.@endTime",
    "crp_lettuce.Intkam.management.@plantDensity",
    # BO params
    "common.CO2dosing.@pureCO2cap",  # e.g. 280
    "comp1.screens.scr1.@enabled",  # e.g. True
    "comp1.screens.scr1.@material",  # e.g. 'scr_Blackout.par'
    "comp1.screens.scr2.@enabled",  # e.g. False
    "comp1.screens.scr2.@material",  # e.g. 'scr_Transparent.par'
    "comp1.illumination.lmp1.@intensity",  # e.g. 100
    "comp1.illumination.lmp1.@maxIglob",  # e.g. 500
]
EP_KEYS = [
    'common.Iglob.Value',
    'common.TOut.Value',
    'common.RHOut.Value',
    'common.Windsp.Value',
]
OP_KEYS = [
    "comp1.Air.T",
    "comp1.Air.RH",
    "comp1.Air.ppm",
    "comp1.PARsensor.Above",
    "comp1.Plant.PlantDensity",
    "comp1.Plant.fractionGroundCover",
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
OP_IN_KEYS = [
    "comp1.Air.T",
    "comp1.Air.RH",
    "comp1.Air.ppm",
    "comp1.PARsensor.Above",
    "comp1.Plant.PlantDensity",
    "comp1.Plant.fractionGroundCover",
]
PL_KEYS = [
    "comp1.Plant.headFW",
    "comp1.Plant.shootDryMatterContent",
    "comp1.Plant.qualityLoss",
]
PL_INIT_VALUE = {
    "comp1.Plant.headFW": 0,
    "comp1.Plant.shootDryMatterContent": 0.055,
    "comp1.Plant.qualityLoss": 0,
}


def get_range(keys, space):
    low_all, high_all = [], []
    for key in keys:
        low, high = space[key]
        low_all.append(np.asarray(low))
        high_all.append(np.asarray(high))
    return np.concatenate(low_all, axis=0), np.concatenate(high_all, axis=0)


def get_norm_data(space=MODEL_PARAM_SPACE):
    return {
        'cp': get_range(CP_KEYS, space),
        'ep': get_range(EP_KEYS, space),
        'pl': get_range(PL_KEYS, space),
        'op': get_range(OP_KEYS, space),
        'op_in': get_range(OP_IN_KEYS, space),
    }
