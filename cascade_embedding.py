import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
def embed_cascades_text():
	model = SentenceTransformer('paraphrase-distilroberta-base-v1')
	directory = os.fsencode(os.path.join(ROOT_DIR, 'twitter-raw-data/Twitter/'))
	cascades_text_embeded = {}
	for cascade_name in os.listdir(directory):
		cascade_dir = os.fsencode(os.path.join(ROOT_DIR, 'twitter-raw-data/Twitter', str(cascade_name).split('\'')[1]))
		cascade_embed = np.array([0.0 for _ in range(768)])
		ctr = 0
		for cascade in os.listdir(cascade_dir):
			fd = open(os.path.join(str(cascade_dir).split('\'')[1], str(cascade).split('\'')[1]), 'r')
			cascade_tweet_text = fd.read()
			fd.close()
			print(cascade_tweet_text)
			cascade_tweet_json = json.loads(cascade_tweet_text)
			cascade_tweet_text = cascade_tweet_json['tweet']['text']
			tweet_embed = model.encode(cascade_tweet_text)
			cascade_embed = np.add(cascade_embed, tweet_embed)
			ctr += 1
		cascade_embed = cascade_embed / ctr
		cascades_text_embeded[str(cascade_name).split('\'')[1]] = cascade_embed
	print(cascade_tweet_text)
	return cascades_text_embeded
			

embed_cascades_text()
