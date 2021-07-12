
import glob, os
from utils import query_simulator, save_json_data, load_json_data
from search import SIM_ID
best_control_file = glob.glob(f'{os.path.dirname(os.path.abspath(__file__))}/best_control_*.json')[0]
best_control = load_json_data(best_control_file)
best_output = query_simulator(best_control, sim_id=SIM_ID)

balance = best_output['stats']['economics']['balance']
print('best netprofit of final submission:', balance)
save_json_data(best_output, f'{os.path.dirname(os.path.abspath(__file__))}/best_output_{balance}.json')