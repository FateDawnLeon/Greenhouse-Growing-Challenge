import glob, os
from utils import query_simulator, save_json_data, load_json_data


SIM_ID = 'C'
BEST_DIR = "/mnt/d/Codes/Greenhouse-Growing-Challenge/collect_data/TPEOpt"
# BEST_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    best_control_file = glob.glob(f'{BEST_DIR}/best_control_*.json')[0]
    best_control = load_json_data(best_control_file)
    best_output = query_simulator(best_control, sim_id=SIM_ID)

    balance = best_output['stats']['economics']['balance']
    print('best netprofit of final submission:', balance)

    prev_outputs = glob.glob(f'{BEST_DIR}/best_output_*.json')
    for f in prev_outputs:
        os.remove(f)
    save_json_data(best_output, f'{BEST_DIR}/best_output_{balance}.json')
    