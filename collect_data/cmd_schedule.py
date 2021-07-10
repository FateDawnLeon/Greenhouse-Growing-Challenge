import os
import time
import pytz
from datetime import date, datetime


def run_BO(simulator='A', num_calls=100, num_init_points=10, dimension_spec='A1', optimizer='gp', precision=0, plan=False):
    today = date.today().isoformat()

    cmd = f'python bo_search.py \
            -L \
            -NC {num_calls} \
            -NI {num_init_points} \
            -DS {dimension_spec} \
            -S {simulator} \
            -O {optimizer} \
            -P {precision} \
            -D data_sample=BO_date={today}_SIM={simulator}_DS={dimension_spec}_OPT={optimizer}_NI={num_init_points}_NC={num_calls}_P={precision}'

    if plan:
        timezone = pytz.timezone('Europe/Amsterdam')
        start_time = datetime(2021, 7, 8, hour=14, tzinfo=timezone)
        while datetime.now(tz=timezone) < start_time:
            print('waiting to start... current time >>>' ,datetime.now(tz=timezone), end='\r')
            time.sleep(1)

    os.system(cmd)


if __name__ == '__main__':
    # run_BO(
    #     simulator='B',
    #     num_calls=300,
    #     num_init_points=30,
    #     dimension_spec='B3',
    #     optimizer='gbrt',
    #     precision=2,
    #     plan=False
    # )
    run_BO(
        simulator='A',
        num_calls=1000,
        num_init_points=100,
        dimension_spec='A4PD2',
        optimizer='gp',
        precision=0,
        plan=False
    )
