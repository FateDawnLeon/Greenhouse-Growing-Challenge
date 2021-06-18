import json
import requests


KEYS = {
	'A': 'C48A-ZRJQ-3wcq-rGuC-mEme',
	'B': 'C48B-PTmQ-89Kx-jqV5-3zRL' 
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/'


def try_on_simulator(json_name, control_json_dir, output_json_dir, key):
	with open(f'{control_json_dir}/{json_name}', 'r') as f:
		control = json.load(f)

	control["key"] = key
	headers = {'ContentType': 'application/json'}
	response = requests.post(URL, data=control, headers=headers, timeout=300)

	output = response.json()
	print(response, output['responsemsg'])

	with open(f'{output_json_dir}/{json_name}', 'w') as f:
		json.dump(output, f)


if __name__ == '__main__':
	import os
	import argparse
	import threadpool

	parser = argparse.ArgumentParser()
	parser.add_argument('-N', '--num-trials', type=int, default=0)
	parser.add_argument('-T', '--num-workers', type=int, default=1)
	parser.add_argument('-S', '--simulator', choices=['A', 'B'], type=str, default='A')
	parser.add_argument('-C', '--clear-invalid-output', action='store_true', default=False)
	parser.add_argument('-F', '--control-json-file', type=str, default=None)
	args = parser.parse_args()

	key = KEYS[args.simulator]
	output_json_dir = f'output_jsons_{args.simulator}'
	os.makedirs(output_json_dir, exist_ok=True)
	
	if args.clear_invalid_output:
		for name in os.listdir(output_json_dir):
			path = os.path.join(output_json_dir, name)
			file_size = os.path.getsize(path) / 1024
			if file_size < 5:
				os.remove(path)
				print(f'{path} has been removed.')

	if args.control_json_file:
		control_json_dir, file_name = os.path.split(args.control_json_file)
		control_json_names = [file_name]
		args.num_trials = 1
	else:
		# only test those control jsons that are not uploaded before
		control_json_dir = 'control_jsons'
		control_json_names = os.listdir(control_json_dir)
		output_json_names = os.listdir(output_json_dir)
		control_json_names = set(control_json_names).difference(output_json_names)

	# if jsons are not enough, just upload all valid ones	
	num_trials = min(args.num_trials, len(control_json_names))

	if num_trials > 0:
		control_json_names = list(control_json_names)[:num_trials]
		
		# using thread pool to automatically send concurrent requests
		pool = threadpool.ThreadPool(args.num_workers)
		func_vars = [([name, control_json_dir, output_json_dir, key], None) for name in control_json_names]
		tasks = threadpool.makeRequests(try_on_simulator, func_vars)

		for task in tasks:
			pool.putRequest(task)

		pool.wait()

	print(f'expected new trials: {args.num_trials}, actual new trials: {num_trials}')
