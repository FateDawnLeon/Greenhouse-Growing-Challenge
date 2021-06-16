import json
import requests


KEY = 'C48A-ZRJQ-3wcq-rGuC-mEme'
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/'


def try_on_simulator(json_name, control_json_dir, output_json_dir):
	with open(f'{control_json_dir}/{json_name}', 'r') as f:
		control = json.load(f)

	control["key"] = KEY
	headers = {'ContentType': 'application/json'}
	response = requests.post(URL, data=control, headers=headers, timeout=300)

	print(response)

	with open(f'{output_json_dir}/{json_name}', 'w') as f:
		output = json.loads(response.text)
		json.dump(output, f)


if __name__ == '__main__':
	import os
	import argparse
	import threadpool

	parser = argparse.ArgumentParser()
	parser.add_argument('-N', '--num-trials', type=int)
	parser.add_argument('-T', '--num-workers', type=int, default=32)
	args = parser.parse_args()

	control_json_dir = 'control_jsons'
	output_json_dir = 'output_jsons'
	os.makedirs(output_json_dir, exist_ok=True)

	# only test those control jsons that are not uploaded before
	control_json_names = os.listdir(control_json_dir)
	output_json_names = os.listdir(output_json_dir)
	control_json_names = set(control_json_names).difference(output_json_names)

	# if jsons are not enough, just upload all valid ones	
	num_trials = min(args.num_trials, len(control_json_names))
	control_json_names = list(control_json_names)[:num_trials]

	# using thread pool to automatically send concurrent requests
	pool = threadpool.ThreadPool(args.num_workers)
	func_vars = [([name, control_json_dir, output_json_dir], None) for name in control_json_names]
	tasks = threadpool.makeRequests(try_on_simulator, func_vars)

	for task in tasks:
		pool.putRequest(task)

	pool.wait()

	print(f'expected new trials: {args.num_trials}, actual new trials: {num_trials}')
