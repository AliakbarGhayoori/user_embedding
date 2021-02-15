import logging
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import pickle
from nose.tools import assert_equal
from kmodes import kprototypes


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='cluster_text.log', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def embed_user_cascades(user_cascades):
	fake_casc_percent = 0
	user_tweets_embed = np.array([0 for _ in range(768)])
	global model
	
	for cascade in user_cascades:
		fake_casc_percent += int(cascade[1])
		user_tweets_embed = np.add(user_tweets_embed, model.encode(cascade[2]))
	
	user_tweets_embed = user_tweets_embed / len(user_cascades)
	fake_casc_percent = fake_casc_percent / len(user_cascades)
	
	return user_tweets_embed, fake_casc_percent


def user_embedding(users_dict: dict):
	global model
	
	user_ids = []
	users_bio = None
	users_tweets_embed = None
	logging.info("start embedding.")
	for user_id, user_info in users_dict.items():
		user_ids += [user_id]
		print(user_info)
		user_bio_embed = model.encode(user_info['profile_features']['description'])
		user_tweets_embed, fake_casc_percent = embed_user_cascades(user_info['cascades_feature'])
		
		if users_bio is None:
			users_bio = [user_bio_embed.tolist()]
			users_tweets_embed = [user_tweets_embed.tolist()]
		else:
			users_bio = np.append(users_bio, [user_bio_embed.tolist()], axis=0)
			users_tweets_embed = np.append(users_tweets_embed, [user_tweets_embed.tolist()], axis=0)
		
	
	pca = PCA(n_components=100)
	users_bio = pca.fit_transform(users_bio)
	users_tweets_embed = pca.fit_transform(users_tweets_embed)
	
	return user_ids, users_bio, users_tweets_embed


def users_clustering(users_ids, users_bio, users_tweet, my_data):
	users_dataset = []
	data_set = {}
	for i in range(len(users_ids)):
		user = users_ids[i]
		user_bio = users_bio[i].tolist()
		user_tweet = users_tweet[i].tolist()
		
		profile_background_tile = 1 if my_data[user]['profile_features']['profile_background_tile'] else 0
		profile_use_background_image = 1 if my_data[user]['profile_features']['profile_use_background_image'] else 0
		screen_name = len(my_data[user]['profile_features']['screen_name'])
		verified = 1 if my_data[user]['profile_features']['verified'] else 0
		statuses_count = my_data[user]['profile_features']['statuses_count']
		favourites_count = my_data[user]['profile_features']['favourites_count']
		has_extended_profile = 1 if my_data[user]['profile_features']['has_extended_profile'] else 0
		friends_count = my_data[user]['profile_features']['friends_count']
		followers_count = my_data[user]['profile_features']['followers_count']
		number_cascades = len(my_data[user]['cascades_feature'])
		users_dataset.append([
								 profile_background_tile,
								 profile_use_background_image,
								 screen_name,
								 verified,
								 statuses_count,
								 favourites_count,
								 has_extended_profile,
								 friends_count,
								 followers_count,
								 number_cascades
							 ] + user_bio + user_tweet)
	
		data_set[i] = user
		
	logging.info("making data matrix finished.")
	
	users_dataset = np.array(users_dataset)
	logging.info('data set created')
	kproto_init = kprototypes.KPrototypes(n_clusters=3600, init="Huang", verbose=2, n_init=1)
	logging.info('go for learning clusters')
	result = kproto_init.fit_predict(users_dataset, categorical=[0, 1, 3, 6])
	logging.info("model fit-predict result:{0}".format(result))
	pickle.dump(result, open('results1_text.p', 'wb'))
	pickle.dump(data_set, open('results11_text.p', 'wb'))
	with open('results1_text.txt', 'w') as f:
		f.write("\n".join(str(result)))


user_dict = pickle.load(open( os.path.join(ROOT_DIR, 'users_data.p'), "rb"))
user_ids, users_bio, users_tweet = user_embedding(user_dict)
users_clustering(user_ids, users_bio, users_tweet, user_dict)

# user_embedding({'12': {'description':'hi to you', 'cascades_feature':[[12, 1, 'this is a test']]},'13': {'description':'hi to me', 'cascades_feature':[[12, 1, 'this is not a test']]}})