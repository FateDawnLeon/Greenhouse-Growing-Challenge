import torch
import datetime


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CITY_NAME = 'Amsterdam'
START_DATE = datetime.date(2021, 3, 4)
MATERIALS = ['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']


# ====================== control param related ======================
CONTROL_KEYS = [
    "simset.@endDate",  # "yyyy-mm-dd" -> fixed
    "comp1.heatingpipes.pipe1.@maxTemp",  # float -> variable
    "comp1.heatingpipes.pipe1.@minTemp",  # float -> variable
    "comp1.heatingpipes.pipe1.@radiationInfluence",  # [float, float] -> variable
    "comp1.setpoints.temp.@heatingTemp",  # float -> variable
    "comp1.setpoints.temp.@ventOffset",  # float -> variable
    "comp1.setpoints.temp.@radiationInfluence",  # [float, float, float] -> variable
    "comp1.setpoints.temp.@PbandVent",  # [float, float, float, float] -> variable
    "comp1.setpoints.ventilation.@startWnd",  # float -> variable
    "comp1.setpoints.ventilation.@winLeeMin",  # float -> variable
    "comp1.setpoints.ventilation.@winLeeMax",  # float -> variable
    "comp1.setpoints.ventilation.@winWndMin",  # float -> variable
    "comp1.setpoints.ventilation.@winWndMax",  # float -> variable
    "common.CO2dosing.@pureCO2cap",  # float -> fixed
    "comp1.setpoints.CO2.@setpoint",  # float -> variable
    "comp1.setpoints.CO2.@setpIfLamps",  # float -> variable
    "comp1.setpoints.CO2.@doseCapacity",  # [float, float, float, float, float, float] -> variable
    "comp1.screens.scr1.@enabled",  # [0/1] -> fixed
    "comp1.screens.scr1.@material",  # [0/1, 0/1, 0/1] -> fixed
    "comp1.screens.scr1.@ToutMax",  # float -> variable
    "comp1.screens.scr1.@closeBelow",  # [float, float, float, float] -> variable
    "comp1.screens.scr1.@closeAbove",  # float -> variable
    "comp1.screens.scr1.@lightPollutionPrevention",  # [0/1] -> variable
    "comp1.screens.scr2.@enabled",  # [0/1] -> fixed
    "comp1.screens.scr2.@material",  # [0/1, 0/1, 0/1] -> fixed
    "comp1.screens.scr2.@ToutMax",  # float -> variable
    "comp1.screens.scr2.@closeBelow",  # [float, float, float, float] -> variable
    "comp1.screens.scr2.@closeAbove",  # float -> variable
    "comp1.screens.scr2.@lightPollutionPrevention",  # [0/1] -> variable
    "comp1.illumination.lmp1.@enabled",  # [0/1] -> variable
    "comp1.illumination.lmp1.@intensity",  # float -> fixed
    "comp1.illumination.lmp1.@hoursLight",  # float -> variable (day-wise)
    "comp1.illumination.lmp1.@endTime",  # float -> variable (day-wise)
    "comp1.illumination.lmp1.@maxIglob",  # float -> variable (day-wise)
    "comp1.illumination.lmp1.@maxPARsum",  # float -> variable (day-wise)
    "crp_lettuce.Intkam.management.@plantDensity",  # float -> variable (day-wise)
]
ENV_KEYS = [
    'common.Iglob.Value',
    'common.TOut.Value',
    'common.RHOut.Value',
    'common.Windsp.Value',
    'common.Economics.PeakHour',
]
OUTPUT_IN_KEYS = [
    "comp1.Air.T",
    "comp1.Air.RH",
    "comp1.Air.ppm",
    "comp1.PARsensor.Above",
    "comp1.Lmp1.ElecUse",
    "comp1.PConPipe1.Value",
    "comp1.McPureAir.Value",
    "comp1.TPipe1.Value",
    "comp1.ConPipes.TSupPipe1",
    "comp1.ConWin.WinLee",
    "comp1.ConWin.WinWnd",
    "comp1.Setpoints.SpHeat",
    "comp1.Setpoints.SpVent",
    "comp1.Scr1.Pos",
    "comp1.Scr2.Pos",
]
OUTPUT_PL_KEYS = [
    "comp1.Plant.headFW",
    "comp1.Plant.shootDryMatterContent",
    "comp1.Plant.fractionGroundCover",
    "comp1.Plant.plantProjection",
    'comp1.Plant.PlantDensity',
]
OUTPUT_KEYS = [
    'comp1.Air.T',
    'comp1.Air.RH',
    'comp1.Air.ppm',
    'comp1.PARsensor.Above',
    'comp1.Plant.headFW',
    'comp1.Plant.shootDryMatterContent',
    'comp1.Lmp1.ElecUse',
    'comp1.PConPipe1.Value',
    'comp1.McPureAir.Value',
    'comp1.Plant.PlantDensity',
    'comp1.TPipe1.Value',
    'comp1.ConPipes.TSupPipe1',
    'comp1.ConWin.WinLee',
    'comp1.ConWin.WinWnd',
    'comp1.Setpoints.SpHeat',
    'comp1.Setpoints.SpVent',
    'comp1.Scr1.Pos',
    'comp1.Scr2.Pos',
    'comp1.Plant.fractionGroundCover',
    'comp1.Plant.plantProjection',
]
ENV_KEYS_TO_INDEX = {key: i for i, key in enumerate(ENV_KEYS)}
OUTPUT_KEYS_TO_INDEX = {key: i for i, key in enumerate(OUTPUT_KEYS)}
OUTPUT_IN_KEYS_TO_INDEX = {key: i for i, key in enumerate(OUTPUT_IN_KEYS)}
OUTPUT_PL_KEYS_TO_INDEX = {key: i for i, key in enumerate(OUTPUT_PL_KEYS)}


# ====================== simulator related ======================
KEYS = {
    'A': 'C48A-ZRJQ-3wcq-rGuC-mEme',
    'B': 'C48B-PTmQ-89Kx-jqV5-3zRL'
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/model.aspx'
SAMPLE_CONTROL_JSON_PATH = './ClimateControlSample.json'
