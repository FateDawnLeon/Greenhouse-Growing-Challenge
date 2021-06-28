import os
import time
import pytz
from datetime import date, datetime


def get_cmd_1():
    simulator = 'B'
    num_calls = 1000
    num_init_points = 100
    random_state = 42
    dimension_spec = 'A'
    optimizer = 'gbrt'
    precision = 1
    today = date.today().isoformat()

    cmd = f'python bo_search.py \
            -S {simulator} \
            -DS {dimension_spec} \
            -NI {num_init_points} \
            -NC {num_calls} \
            -RS {random_state} \
            -P {precision} \
            -D bo_data_{today}_SIM={simulator}__DS={dimension_spec}_OPT={optimizer}_NI={num_init_points}_NC={num_calls}_RS={random_state}_P={precision}'
    
    return cmd


if __name__ == '__main__':
    cmd = get_cmd_1()
    print(cmd)

    timezone = pytz.timezone('Europe/Amsterdam')
    start_time = datetime(2021, 6, 28, hour=14, tzinfo=timezone)
    while datetime.now(tz=timezone) < start_time:
        print('waiting to start... current time >>>' ,datetime.now(tz=timezone), end='\r')
        time.sleep(1)

    os.system(cmd)
