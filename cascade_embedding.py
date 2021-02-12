import os
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
def traverse_file():
	directory = os.fsencode(os.path.join(ROOT_DIR, '../twitter-raw-data/Twitter/'))
	for cascade_name in os.listdir(directory):
		cascade_dir = os.fsencode(os.path.join(ROOT_DIR, '../twitter-raw-data/Twitter', cascade_name))
		print(cascade_dir)

traverse_file()
