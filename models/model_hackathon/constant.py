import torch
import datetime
import numpy as np

from collections import OrderedDict
from gym.spaces import Box

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CITY_NAME = 'Amsterdam'
START_DATE = datetime.date(2021, 3, 4)
MATERIALS = ['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']

# ====================== control param related ======================

# map KEYS -> (USE_RL, SIZE)
# CONTROL_INFO = OrderedDict([
#     ("end", (True, 1)),
#     ("comp1.heatingpipes.pipe1.@maxTemp", (False, 1)),
#     ("comp1.heatingpipes.pipe1.@minTemp", (False, 1)),
#     ("comp1.heatingpipes.pipe1.@radiationInfluence", (False, 2)),
#     ("comp1.setpoints.temp.@heatingTemp", (True, 2)),
#     ("comp1.setpoints.temp.@ventOffset", (True, 1)),
#     ("comp1.setpoints.temp.@radiationInfluence", (False, 3)),
#     ("comp1.setpoints.temp.@PbandVent", (False, 4)),
#     ("comp1.setpoints.ventilation.@startWnd", (True, 1)),
#     ("comp1.setpoints.ventilation.@winLeeMin", (False, 1)),
#     ("comp1.setpoints.ventilation.@winLeeMax", (False, 1)),
#     ("comp1.setpoints.ventilation.@winWndMin", (False, 1)),
#     ("comp1.setpoints.ventilation.@winWndMax", (False, 1)),
#     ("common.CO2dosing.@pureCO2cap", (False, 1)),
#     ("comp1.setpoints.CO2.@setpoint", (True, 2)),
#     ("comp1.setpoints.CO2.@setpIfLamps", (False, 1)),
#     ("comp1.setpoints.CO2.@doseCapacity", (False, 6)),
#     ("comp1.screens.scr1.@enabled", (False, 1)),
#     ("comp1.screens.scr1.@material", (False, 3)),
#     ("comp1.screens.scr1.@ToutMax", (True, 1)),
#     ("comp1.screens.scr1.@closeBelow", (False, 4)),
#     ("comp1.screens.scr1.@closeAbove", (False, 1)),
#     ("comp1.screens.scr1.@lightPollutionPrevention", (False, 1)),
#     ("comp1.screens.scr2.@enabled", (False, 1)),
#     ("comp1.screens.scr2.@material", (False, 3)),
#     ("comp1.screens.scr2.@ToutMax", (True, 1)),
#     ("comp1.screens.scr2.@closeBelow", (False, 4)),
#     ("comp1.screens.scr2.@closeAbove", (False, 1)),
#     ("comp1.screens.scr2.@lightPollutionPrevention", (False, 1)),
#     ("comp1.illumination.lmp1.@enabled", (False, 1)),
#     ("comp1.illumination.lmp1.@intensity", (False, 1)),
#     ("comp1.illumination.lmp1.@hoursLight", (True, 1)),
#     ("comp1.illumination.lmp1.@endTime", (True, 1)),
#     ("comp1.illumination.lmp1.@maxIglob", (False, 1)),
#     ("comp1.illumination.lmp1.@maxPARsum", (False, 1)),
#     ("crp_lettuce.Intkam.management.@plantDensity", (True, 1))
# ])


CONTROL_RL = OrderedDict([
    ("comp1.setpoints.temp.@heatingTemp", 2),  # temp_night, temp_day
    ("comp1.setpoints.temp.@ventOffset", 1),
    ("comp1.setpoints.ventilation.@startWnd", 1),
    ("comp1.setpoints.CO2.@setpoint", 2),
    ("comp1.screens.scr1.@ToutMax", 1),
    ("comp1.screens.scr1.@closeBelow", 1),
    ("comp1.screens.scr1.@closeAbove", 1),
    ("comp1.screens.scr2.@ToutMax", 1),
    ("comp1.screens.scr2.@closeBelow", 1),
    ("comp1.screens.scr2.@closeAbove", 1),
    ("comp1.illumination.lmp1.@hoursLight", 1),
    ("comp1.illumination.lmp1.@endTime", 1),
    ("comp1.illumination.lmp1.@hoursLight", 1),  # [0, 10]
    ("crp_lettuce.Intkam.management.@plantDensity", 2)  # (value, change)
])

CONTROL_BO = OrderedDict([
    ("simset.@startDate", 1),
    ("common.CO2dosing.@pureCO2cap", 1),
    ("comp1.screens.scr1.@enabled", 1),
    ("comp1.screens.scr1.@material", 3),
    ("comp1.screens.scr2.@enabled", 1),
    ("comp1.screens.scr2.@material", 3),
    ("comp1.illumination.lmp1.@intensity", 1),
    ("comp1.illumination.lmp1.@maxIglob", 1)
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
    "crp_lettuce.Intkam.management.@plantDensity",  # e.g. "1 90; 7 60; 14 40; 21 30; 28 20; 34 15"
]

CONTROL_OTHER = [
    "comp1.screens.scr1.@lightPollutionPrevention",  # scr1_material == "scr_Blackout.par"
    "comp1.screens.scr2.@lightPollutionPrevention",  # scr2_material == "scr_Blackout.par"
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

# ====================== runtime related ======================
dtype = np.float32
CP_KEY_SPACES = {
    "comp1.setpoints.temp.@heatingTemp": Box(
        low=np.asarray([5, 15], dtype=dtype), high=np.asarray([15, 30], dtype=dtype)),
    "comp1.setpoints.temp.@ventOffset": Box(
        low=np.asarray([0], dtype=dtype), high=np.asarray([5], dtype=dtype)),
    "comp1.setpoints.ventilation.@startWnd": Box(
        low=np.asarray([0], dtype=dtype), high=np.asarray([50], dtype=dtype)),
    "comp1.setpoints.CO2.@setpoint": Box(
        low=np.asarray([0, 400], dtype=dtype), high=np.asarray([400, 1200], dtype=dtype)),
    "comp1.screens.scr1.@ToutMax": Box(
        low=np.asarray([-20], dtype=dtype), high=np.asarray([30], dtype=dtype)),
    "comp1.screens.scr1.@closeBelow": Box(
        low=np.asarray([0], dtype=dtype), high=np.asarray([200], dtype=dtype)),
    "comp1.screens.scr1.@closeAbove": Box(
        low=np.asarray([500], dtype=dtype), high=np.asarray([1500], dtype=dtype)),
    "comp1.screens.scr2.@ToutMax": Box(
        low=np.asarray([-20], dtype=dtype), high=np.asarray([30], dtype=dtype)),
    "comp1.screens.scr2.@closeBelow": Box(
        low=np.asarray([0], dtype=dtype), high=np.asarray([200], dtype=dtype)),
    "comp1.screens.scr2.@closeAbove": Box(
        low=np.asarray([500], dtype=dtype), high=np.asarray([1500], dtype=dtype)),
    "comp1.illumination.lmp1.@hoursLight": Box(
        low=np.asarray([0], dtype=dtype), high=np.asarray([18], dtype=dtype)),
    "comp1.illumination.lmp1.@endTime": Box(
        low=np.asarray([18], dtype=dtype), high=np.asarray([20], dtype=dtype)),
}
EP_KEY_SPACES = {
    'common.Iglob.Value': (0, 1000),
    'common.TOut.Value': (-10, 40),
    'common.RHOut.Value': (0, 100),
    'common.Windsp.Value': (0, 10),
}
OP_KEY_SPACES = {
    'comp1.Air.T': (-10, 40),
    'comp1.Air.RH': (0, 100),
    'comp1.Air.ppm': (0, 2000),
    'comp1.PARsensor.Above': (0, 2000),
    'comp1.Plant.PlantDensity': (0, 100),
    "comp1.Plant.fractionGroundCover": (0, 1),
    'comp1.TPipe1.Value': (0, 80),
    'comp1.ConPipes.TSupPipe1': (0, 80),
    'comp1.PConPipe1.Value': (0, 200),
    'comp1.ConWin.WinLee': (0, 100),
    'comp1.ConWin.WinWnd': (0, 100),
    'comp1.Setpoints.SpHeat': (0, 30),
    'comp1.Setpoints.SpVent': (0, 30),
    'comp1.Scr1.Pos': (0, 1),
    'comp1.Scr2.Pos': (0, 1),
    'comp1.Lmp1.ElecUse': (0, 10),
    'comp1.McPureAir.Value': (0, 1e-5),
}
PL_KEY_SPACES = {
    'comp1.Plant.headFW': (0, 1000),
    'comp1.Plant.shootDryMatterContent': (0, 0.1),
    'comp1.Plant.qualityLoss': (0, 100),
}
OP_IN_KEYS = [
    'comp1.Air.T',
    'comp1.Air.RH',
    'comp1.Air.ppm',
    'comp1.PARsensor.Above',
    'comp1.Plant.PlantDensity',
    "comp1.Plant.fractionGroundCover",
]


def get_low_high(space, keys):
    lows, highs = [], []
    for key in keys:
        range = space[key]
        low = range.low if isinstance(range, Box) else np.asarray([range[0]], dtype=dtype)
        high = range.high if isinstance(range, Box) else np.asarray([range[1]], dtype=dtype)
        lows.append(low)
        highs.append(high)
    return np.concatenate(lows, axis=0), np.concatenate(highs, axis=0)


def get_norm_data():
    return {
        'cp': get_low_high(CP_KEY_SPACES, CP_KEY_SPACES.keys()),
        'ep': get_low_high(EP_KEY_SPACES, EP_KEY_SPACES.keys()),
        'op': get_low_high(OP_KEY_SPACES, OP_KEY_SPACES.keys()),
        'pl': get_low_high(PL_KEY_SPACES, PL_KEY_SPACES.keys()),
        'op_in': get_low_high(OP_KEY_SPACES, OP_IN_KEYS),
    }

# ====================== model related ======================
CLIMATE_MODEL_PATH = ''  # TODO
PLANT_MODEL_PATH = ''  # TODO
