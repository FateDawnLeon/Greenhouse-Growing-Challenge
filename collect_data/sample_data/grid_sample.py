from utils import ControlParams


temp_high = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
temp_low = [8, 10, 12, 14, 16]
temp = [18, 19, 20, 21, 22, 23, 24, 25]
dark_time = [1, 2, 3, 4, 6]



search_space_A = {
    # end date control
    "duration": [30, 35, 40, 45, 50],
    
    # temperature control
    "heatingTemp": [
        {"01-01": {"r-1": t_low,"r+1": t_high, "s-1": t_high, "s+1": t_low}} for t_low in temp_low for t_high in temp_high if t_low <= t_high
    ], 

    # CO2 control
    "pureCO2Cap": [100, 150, 200],
    "setpoint": [400, 500, 600, 700],

    # lighting time and intensity control
    "hoursLight": [24 - dt for dt in dark_time],
    "endTime": [18, 20, 22, 24],

    # ventilation control
    "startWnd": [20, 30, 40, 50],
}


search_space_B = {
    # end date control
    # "duration": [30, 35, 40, 45, 50],
    
    # temperature control
    "heatingTemp": [
        # {"01-01": {"r-1": t_low,"r+1": t_high, "s-1": t_high, "s+1": t_low}} for t_low in temp_low for t_high in temp_high if t_low <= t_high
        {"01-01": {"8.0":t}} for t in temp
    ], 

    # CO2 control
    # "pureCO2Cap": [100, 150, 200],
    "setpoint": [400, 500, 600, 700, 800, 900, 1000, 1100, 1200],

    # lighting time and intensity control
    "hoursLight": [24 - dt for dt in dark_time],
    # "endTime": [18, 20, 22, 24],

    # ventilation control
    # "startWnd": [20, 30, 40, 50],
}


def grid_search_A():
    search_space = search_space_A

    cnt = 0
    for duration in search_space['duration']:
        for heatingTemp in search_space['heatingTemp']:
            for pureCO2Cap in search_space['pureCO2Cap']:
                for setpoint in search_space['setpoint']:
                    for hoursLight in search_space['hoursLight']:
                        for endTime in search_space['endTime']:
                            for startWnd in search_space['startWnd']:
                                CP = ControlParams(initialize_with_sample=True)
                                
                                CP.set_end_date(duration=duration)
                                CP.set_temperature(heatingTemp=heatingTemp)
                                CP.set_CO2(pureCO2cap=pureCO2Cap, setpoint=setpoint)
                                CP.set_illumination(enabled=True, hoursLight=hoursLight, endTime=endTime)
                                CP.set_ventilation(startWnd=startWnd)

                                # hashcode = hex(hash(str(CP)))
                                # CP.save_as_json(f'control_jsons/{hashcode}')
                                # print(hashcode)

                                cnt += 1
                                print(cnt)

    print('total number:', cnt)


def grid_search_B():
    search_space = search_space_B

    cnt = 0
    for heatingTemp in search_space['heatingTemp']:
        for setpoint in search_space['setpoint']:
            for hoursLight in search_space['hoursLight']:
                CP = ControlParams(initialize_with_sample=True)
                CP.set_temperature(heatingTemp=heatingTemp)
                CP.set_CO2(setpoint=setpoint)
                CP.set_illumination(hoursLight=hoursLight)

                hashcode = hex(hash(CP))
                import os
                os.makedirs('control_jsons_grid', exist_ok=True)
                CP.save_as_json(f'control_jsons_grid/{hashcode}')

                cnt += 1
                print(f'Control parameters {cnt} generated: {hashcode}')


if __name__ == '__main__':
    grid_search_B()
