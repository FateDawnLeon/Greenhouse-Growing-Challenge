import os
import time
import pytz
from datetime import date, datetime


def run_simB_BO():
    simulator = 'B'
    num_calls = 1000
    num_init_points = 100
    dimension_spec = 'BB'
    optimizer = 'gbrt'
    precision = 1
    today = date.today().isoformat()

    cmd = f'python bo_search.py \
            -NC {num_calls} \
            -NI {num_init_points} \
            -DS {dimension_spec} \
            -S {simulator} \
            -O {optimizer} \
            -L \
            -P {precision} \
            -D data_sample=BO_date={today}_SIM={simulator}_DS={dimension_spec}_OPT={optimizer}_NI={num_init_points}_NC={num_calls}_P={precision}'

    timezone = pytz.timezone('Europe/Amsterdam')
    start_time = datetime(2021, 7, 5, hour=14, tzinfo=timezone)
    while datetime.now(tz=timezone) < start_time:
        print('waiting to start... current time >>>' ,datetime.now(tz=timezone), end='\r')
        time.sleep(1)

    os.system(cmd)


def run_simA_BO():
    simulator = 'A'
    num_calls = 5000
    num_init_points = 500
    dimension_spec = 'AA'
    optimizer = 'gbrt'
    precision = 0
    today = date.today().isoformat()

    cmd = f'python bo_search.py \
            -S {simulator} \
            -DS {dimension_spec} \
            -NI {num_init_points} \
            -NC {num_calls} \
            -P {precision} \
            -L \
            -D bo_data_{today}_SIM={simulator}__DS={dimension_spec}_OPT={optimizer}_NI={num_init_points}_NC={num_calls}_P={precision}'

    os.system(cmd)


if __name__ == '__main__':
    run_simB_BO()
    # run_simA_BO()
