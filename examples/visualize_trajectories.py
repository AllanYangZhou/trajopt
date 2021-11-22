"""
Helper script for visualizing optimized trajectories
Expects trajectories in trajopt format
See: https://github.com/aravindr93/trajopt.git
"""

import os
import glob
import pickle
import click 

DESC = '''
Helper script to visualize optimized trajectories (list of trajectories in trajopt format).\n
USAGE:\n
    $ python viz_trajectories.py --file path_to_file.pickle --repeat 100\n
'''
@click.command(help=DESC)
@click.option('--file', type=str, help='pickle file with trajectories', required= True)
@click.option('--repeat', type=int, help='number of times to play trajectories', default=1)
def main(file, repeat):
	os.mkdir("viz_vids")
	fnames = glob.glob(file)
	trajectories = []
	for fname in fnames:
		trajectories = trajectories + pickle.load(open(fname, 'rb'))
	for i in range(repeat):
		for j, traj in enumerate(trajectories):
			traj.render_result(os.path.join("viz_vids", f"traj{i}_repeat{j}.mp4"))

if __name__ == '__main__':
	main()