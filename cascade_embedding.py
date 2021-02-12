import os
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
def traverse_file():
	directory = os.fsencode(os.path.join(ROOT_DIR, 'twitter-raw-data/Twitter/'))
	for cascade_name in os.listdir(directory):
		cascade_dir = os.fsencode(os.path.join(ROOT_DIR, 'twitter-raw-data/Twitter', str(cascade_name).split('\'')[1]))
		for cascade in os.listdir(cascade_dir):
			fd = open(os.path.join(str(cascade_dir).split('\'')[1], str(cascade).split('\'')[1]), 'r')
			file_text = fd.read()
			fd.close()
			print(file_text)
			

traverse_file()
