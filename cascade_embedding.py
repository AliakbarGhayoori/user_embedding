import json
import logging
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='pca_test.log', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

def embed_cascades_text():
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
	
	
def embed_user_cascades(user_cascades):
	fake_casc_percent = 0
	user_tweets_embed = np.array([0 for _ in range(768)])
	global model
	
	for cascade in user_cascades:
		fake_casc_percent += cascade[1]
		user_tweets_embed = np.add(user_tweets_embed, model.encode(cascade[2]))
		
	user_tweets_embed = user_tweets_embed / len(user_cascades)
	fake_casc_percent = fake_casc_percent / len(user_cascades)
	
	return user_tweets_embed, fake_casc_percent

def user_embedding(users_dict : dict):
	global model
	users_bio = pd.DataFrame()
	
	user_ids = []
	users_bio = None
	users_tweets_embed = None
	logging.info("start embedding.")
	for user_id, user_info in users_dict.items():
		user_ids += user_id
		user_bio_embed = model.encode(user_info['description'])
		user_tweets_embed, fake_casc_percent = embed_user_cascades(user_info['cascades_feature'])
		
		if users_bio is None:
			users_bio  = [user_bio_embed.tolist()]
			users_tweets_embed = [user_tweets_embed.tolist()]
		else:
			users_bio = np.append(users_bio, [user_bio_embed.tolist()], axis=0)
			users_tweets_embed = np.append(users_tweets_embed, [user_tweets_embed.tolist()], axis = 0)
		pca = PCA(n_components=200)
		users_bio = pca.fit_transform(users_bio)
		users_tweets_embed = pca.fit_transform(users_tweets_embed)
		
		logging.info("users bio: {0}".format(users_bio))
		logging.info("users tweets embed".format(users_tweets_embed))
		
user_dict = pickle.load(open( os.path.join(ROOT_DIR, '../../Representations/users_data.p'), "rb"))
user_embedding(user_dict)
		
	
#user_embedding({'12': {'description':'hi to you', 'cascades_feature':[[12, 1, 'this is a test']]},'13': {'description':'hi to me', 'cascades_feature':[[12, 1, 'this is not a test']]}})
		