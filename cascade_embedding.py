import os
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
def traverse_file():
	directory = os.fsencode(os.path.join(ROOT_DIR, '../twitter-raw-data/Twitter/'))
	for cascade_dir in os.listdir(directory):
		print(cascade_dir)