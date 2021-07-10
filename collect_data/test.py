import numpy as np
import random

def add_plantDensity():
    start_density_range = np.arange(80, 91, 5)  # 80,85,90
    end_density_range = np.arange(5, 16, 5)  # 5,10,15
    skip_day_range = np.arange(5, 11, 1)  # 5,6,7,8,9,10
    change_density_range = np.arange(5, 36, 5)  # 5,10,15,20,25,30,35

    max_days = 38
    control_densitys = []

    for i in range(20000):
        # "1 90; 7 60; 14 40; 21 30; 28 20; 34 15"
        start_density = random.choice(start_density_range)
        end_density = random.choice(end_density_range)
            
        days = 1
        density = start_density
        
        skip_days = []
        change_densitys = []
        while True:
            skip_day = random.choice(skip_day_range)
            change_density = random.choice(change_density_range)
            days = days+skip_day
            if days>max_days: break
            density = density - change_density
            if density<end_density: break
            skip_days.append(skip_day)
            change_densitys.append(change_density)
        change_densitys.sort(reverse=True)

        days = 1
        density = start_density
        control_density =  f'{days} {density}'
        for i in range(len(skip_days)):
            days = days + skip_days[i]
            density = density - change_densitys[i]
            control_density = f'{control_density}; {days} {density}'

        if density in end_density_range:
            control_densitys.append(control_density)
    print('num of control densitys:', len(control_densitys))
    unique_densitys = set(control_densitys)
    print('num of unique densitys:', len(unique_densitys))

add_plantDensity()