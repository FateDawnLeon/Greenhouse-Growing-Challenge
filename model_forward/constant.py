import datetime
from astral.geocoder import lookup, database


CONTROL_KEYS = [
    "simset.@endDate", # "yyyy-mm-dd" -> fixed
    "comp1.heatingpipes.pipe1.@maxTemp", # float -> variable
    "comp1.heatingpipes.pipe1.@minTemp", # float -> variable 
    "comp1.heatingpipes.pipe1.@radiationInfluence", # [float, float] -> variable
    "comp1.setpoints.temp.@heatingTemp", # float -> variable
    "comp1.setpoints.temp.@ventOffset", # float -> variable
    "comp1.setpoints.temp.@radiationInfluence", # [float, float, float] -> variable
    "comp1.setpoints.temp.@PbandVent", # [float, float, float, float] -> variable
    "comp1.setpoints.ventilation.@startWnd", # float -> variable
    "comp1.setpoints.ventilation.@winLeeMin", # float -> variable
    "comp1.setpoints.ventilation.@winLeeMax", # float -> variable
    "comp1.setpoints.ventilation.@winWndMin", # float -> variable
    "comp1.setpoints.ventilation.@winWndMax", # float -> variable
    "common.CO2dosing.@pureCO2cap", # float -> fixed
    "comp1.setpoints.CO2.@setpoint", # float -> variable
    "comp1.setpoints.CO2.@setpIfLamps", # float -> variable
    "comp1.setpoints.CO2.@doseCapacity", # [float, float, float, float, float, float] -> variable
    "comp1.screens.scr1.@enabled", # [0/1] -> fixed
    "comp1.screens.scr1.@material", # [0/1, 0/1, 0/1] -> fixed
    "comp1.screens.scr1.@ToutMax", # float -> variable
    "comp1.screens.scr1.@closeBelow", # [float, float, float, float] -> variable
    "comp1.screens.scr1.@closeAbove", # float -> variable
    "comp1.screens.scr1.@lightPollutionPrevention", # [0/1] -> variable
    "comp1.screens.scr2.@enabled", # [0/1] -> fixed
    "comp1.screens.scr2.@material", # [0/1, 0/1, 0/1] -> fixed
    "comp1.screens.scr2.@ToutMax", # float -> variable
    "comp1.screens.scr2.@closeBelow", # [float, float, float, float] -> variable
    "comp1.screens.scr2.@closeAbove", # float -> variable
    "comp1.screens.scr2.@lightPollutionPrevention", # [0/1] -> variable
    "comp1.illumination.lmp1.@enabled", # [0/1] -> variable
    "comp1.illumination.lmp1.@intensity", # float -> fixed
    "comp1.illumination.lmp1.@hoursLight", # float -> variable (day-wise)
    "comp1.illumination.lmp1.@endTime", # float -> variable (day-wise)
    "comp1.illumination.lmp1.@maxIglob", # float -> variable (day-wise)
    "comp1.illumination.lmp1.@maxPARsum", # float -> variable (day-wise)
    "crp_lettuce.Intkam.management.@plantDensity", # float -> variable (day-wise)
]

ENV_KEYS = [
    'common.Iglob.Value',
    'common.TOut.Value',
    'common.RHOut.Value',
    'common.Windsp.Value',
    'common.Economics.PeakHour', # related to variable cost
]

OUTPUT_KEYS_RANGE = {
    #================= no need to predict =================
    # Greenhouse Env Param: directly related to FW and DMC
    'comp1.Air.T': [0, 30], # lower=0, upper=30
    'comp1.Air.RH': [0, 100], # lower=0, upper=100
    'comp1.Air.ppm': [0, 1200], # lower=0, upper=1200
    'comp1.PARsensor.Above': [0, 1200], # lower=0, upper=1200
    # Crop Growth Param: directly realted to gain
    'comp1.Plant.headFW': [0, 500], # lower=0, upper=500
    'comp1.Plant.shootDryMatterContent': [0, 0.1], # lower=0
    # Variable Cost Param: related to variable cost
    'comp1.Lmp1.ElecUse': [0, 100], # lower=0
    'comp1.PConPipe1.Value': [0, 200], # lower=0, upper=200
    'comp1.McPureAir.Value': [0, 1e-5], # lower=0
    
    #================= no need to predict =================
    # Plant Density: directly realted to fixed cost
    'comp1.Plant.PlantDensity': [0, 100], # lower=0, upper=100
    # Other Stats
    'comp1.TPipe1.Value': [0, 60], # lower=0, upper=60
    'comp1.ConPipes.TSupPipe1': [0, 60], # lower=0, upper=60
    'comp1.ConWin.WinLee': [0, 100], # lower=0, upper=100
    'comp1.ConWin.WinWnd': [0, 100], # lower=0, upper=100
    'comp1.Setpoints.SpHeat': [0, 30], # lower=0, upper=30
    'comp1.Setpoints.SpVent': [0, 30], # lower=0, upper=30
    'comp1.Scr1.Pos': [0, 1], # lower=0, upper=1
    'comp1.Scr2.Pos': [0, 1], # lower=0, upper=1
    'comp1.Plant.fractionGroundCover': [0, 1], # lower=0, upper=1
    'comp1.Plant.plantProjection': [0, 0.1], # lower=0

    # ===================== Assumption =====================
    # CP + EP -> GEP
    # CP + EP -> VCP
    # CP + GEP -> CGP
    # CP -> PD
    # CP + EP + GEP -> OS
}

OUTPUT_KEYS = list(OUTPUT_KEYS_RANGE.keys())

KEYS = {
    'A': 'C48A-ZRJQ-3wcq-rGuC-mEme',
    'B': 'C48B-PTmQ-89Kx-jqV5-3zRL' 
}

URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/model.aspx'

SAMPLE_CONTROL_JSON_PATH = '../collect_data/ClimateControlSample.json'

START_DATE = datetime.date(2021, 3, 4)

CITY = lookup('Amsterdam', database())

COMMON_DATA_DIR = '../collect_data/common' #TODO: change this to your own directory

EP_PATHS = {sim_id: f'{COMMON_DATA_DIR}/EP-SIM={sim_id}.npy' for sim_id in ['A', 'B', 'C', 'D']}

INIT_STATE_PATHS = {sim_id: f'{COMMON_DATA_DIR}/OP1-POOL-SIM={sim_id}.npy'  for sim_id in ['A', 'B', 'C', 'D']}

EP_PATH = EP_PATHS['A'] # TODO: change the simulator id as your will

INIT_STATE_PATH = INIT_STATE_PATHS['A'] # TODO: change the simulator id as your will

STATE_DICT_PATH = '/home/leondawn/AGC-Model-Weights/basic_mlp_step=6000.pth' #TODO: new model path may be used
