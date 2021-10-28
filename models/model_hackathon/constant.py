from collections import OrderedDict

import torch
import datetime
import numpy as np

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
    ("end", 1),  # {true, false}
    ("comp1.setpoints.temp.@heatingTemp", 2),  # night:[8, 12], day:[20, 25]
    ("comp1.setpoints.temp.@ventOffset", 1),
    ("comp1.setpoints.ventilation.@startWnd", 1),
    ("comp1.setpoints.CO2.@setpoint", 2),
    ("comp1.screens.scr1.@ToutMax", 1),
    ("comp1.screens.scr1.@closeBelow", 1),
    ("comp1.screens.scr1.@closeAbove", 1),
    ("comp1.screens.scr2.@ToutMax", 1),
    ("comp1.screens.scr2.@closeBelow", 1),
    ("comp1.screens.scr2.@closeAbove", 1),
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

CONTROL_OTHER = OrderedDict([
    ("comp1.screens.scr1.@lightPollutionPrevention", 1),  # depend on scr1_material
    ("comp1.screens.scr2.@lightPollutionPrevention", 1),  # depend on scr2_material
    ("comp1.illumination.lmp1.@endTime", 1),  # depend on sunset and lighthour
])

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
    'D': 'C48B-PTmQ-89Kx-jqV5-3zRL'
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/model.aspx'
SAMPLE_CONTROL_JSON_PATH = './ClimateControlSample.json'

# ====================== data related ======================
TRACES_DIR = ''  # TODO

# ====================== runtime related ======================
CP_KEY_RANGES = {
    "end": [0, 1],  # end
    "comp1.setpoints.temp.@heatingTemp.night": [10, 15],
    "comp1.setpoints.temp.@heatingTemp.day": [15, 30],
    "comp1.setpoints.temp.@ventOffset": [0, 5],
    "comp1.setpoints.ventilation.@startWnd": [0, 50],
    "comp1.setpoints.CO2.@setpoint.night": [200, 800],
    "comp1.setpoints.CO2.@setpoint.day": [800, 1200],
    "comp1.screens.scr1.@ToutMax": [-20, 30],
    "comp1.screens.scr1.@closeBelow": [0, 200],
    "comp1.screens.scr1.@closeAbove": [1000, 1500],
    "comp1.screens.scr2.@ToutMax": [-20, 30],
    "comp1.screens.scr2.@closeBelow": [0, 200],
    "comp1.screens.scr2.@closeAbove": [1000, 1500],
    "comp1.illumination.lmp1.@hoursLight": [0, 10],
    "crp_lettuce.Intkam.management.@plantDensity": [0, 50],
}
EP_KEY_RANGES = {
    'common.Iglob.Value': (0, 1000),
    'common.TOut.Value': (-10, 40),
    'common.RHOut.Value': (0, 100),
    'common.Windsp.Value': (0, 10)
}
OP_KEY_RANGES = {
    'comp1.Air.T': (-10, 40),
    'comp1.Air.RH': (0, 100),
    'comp1.Air.ppm': (0, 2000),
    'comp1.PARsensor.Above': (0, 2000),
    'comp1.Plant.PlantDensity': (0, 100),
    'comp1.TPipe1.Value': (0, 80),
    'comp1.ConPipes.TSupPipe1': (0, 80),
    'comp1.PConPipe1.Value': (0.0, 200),
    'comp1.ConWin.WinLee': (0, 100.0),
    'comp1.ConWin.WinWnd': (0, 100.0),
    'comp1.Setpoints.SpHeat': (0, 30),
    'comp1.Setpoints.SpVent': (0, 30),
    'comp1.Scr1.Pos': (0, 1),
    'comp1.Scr2.Pos': (0, 1),
    'comp1.Lmp1.ElecUse': (0, 10),
    'comp1.McPureAir.Value': (0, 1e-5),
}
OP_IN_KEY_RANGES = {
    'comp1.Air.T': (-10, 40),
    'comp1.Air.RH': (0, 100),
    'comp1.Air.ppm': (0, 2000),
    'comp1.PARsensor.Above': (0, 2000),
    'comp1.Plant.PlantDensity': (0, 100)
}
PL_KEY_RANGES = {
    'comp1.Plant.headFW': (0, 1000),
    'comp1.Plant.shootDryMatterContent': (0, 0.1),
    'comp1.Plant.qualityLoss': (0, 15)
}

def get_low_high(key_ranges):
    return (
        np.asarray([v[0] for v in key_ranges.values()], dtype=np.float32),
        np.asarray([v[1] for v in key_ranges.values()], dtype=np.float32),
    )

CLIMATE_NORM_DATA = {
    'cp': get_low_high(CP_KEY_RANGES),
    'ep': get_low_high(EP_KEY_RANGES),
    'op': get_low_high(OP_KEY_RANGES),
}
PLANT_NORM_DATA = {
    'op_in': get_low_high(OP_IN_KEY_RANGES),
    'op_pl': get_low_high(PL_KEY_RANGES),
}

# ====================== model related ======================
CLIMATE_MODEL_PATH = ''  # TODO
PLANT_MODEL_PATH = ''  # TODO

# ===================== BO hyper-parameters ======================
# TODO: temporary, this needs to be updated in BO loop somehow
INIT_PLANT_DENSITY = 90.0
BO_CONTROLS = {
    'common.CO2dosing.@pureCO2cap': 280,
    'comp1.illumination.lmp1.@intensity': 120,
    'comp1.screens.scr1.@enabled': True,
    'comp1.screens.scr2.@enabled': True
}
