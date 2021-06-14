import os
import json
import requests


KEY = 'C48A-ZRJQ-3wcq-rGuC-mEme'


def try_on_simulator(json_name, json_dir, save_dir):
	with open(f'{json_dir}/{json_name}', 'r') as f:
		control = json.load(f)

	control["key"] = KEY
	headers = {
		'ContentType': 'application/json'
	}

	url = 'https://www.digigreenhouse.wur.nl/Kasprobeta/'

	response = requests.post(url, data=control, headers=headers, timeout=300)

	print(response)

	os.makedirs(save_dir, exist_ok=True)

	with open(f'{save_dir}/output-cp={json_name}.json', 'w') as f:
		output = json.loads(response.text)
		json.dump(output, f)


if __name__ == '__main__':
	json_names = os.listdir('control_jsons')

	# print(json_names)

	for json_name in json_names[:1]:
		try_on_simulator(json_name, 'control_jsons', 'output_jsons')
	