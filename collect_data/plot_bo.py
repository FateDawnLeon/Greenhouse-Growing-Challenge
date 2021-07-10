import os
import matplotlib.pyplot as plt
from data import load_json_data

output_dir = 'data_sample=BO_date=2021-07-08_SIM=A_DS=A4BSTPD2_OPT=gbrt_NI=100_NC=1000_P=0/outputs'

profits = []
for name in os.listdir(output_dir):
    output = load_json_data(f'{output_dir}/{name}')
    profits.append(output['stats']['economics']['balance'])

plt.figure(dpi=100)
plt.hist(profits, bins=100, density=True)
plt.title(output_dir.split('/')[-2])
plt.xlabel('netprofit')
plt.ylabel('prob-density')
plt.show()