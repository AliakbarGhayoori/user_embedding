import logging
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import pickle
from nose.tools import assert_equal
from kmodes import kprototypes
import pandas as pd
from sklearn.neighbors._dist_metrics import DistanceMetric
from sklearn.cluster import AgglomerativeClustering, KMeans
import gower
from sklearn.preprocessing import MaxAbsScaler

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
    logging.info("start embedding.")
    user_ids = []
    users_bio = None
    users_tweets_embed = None
    logging.info("start embedding.")
    
    ctr = 0
    for user_id, user_info in users_dict.items():
        if ctr >= 5000:
            break
        ctr += 1
        
        
        user_ids += [user_id]
        user_bio_embed = model.encode(
            user_info['profile_features']['description'])
        user_tweets_embed, fake_casc_percent = embed_user_cascades(
            user_info['cascades_feature'])

        if users_bio is None:
            users_bio = [user_bio_embed.tolist()]
            users_tweets_embed = [user_tweets_embed.tolist()]
        else:
            users_bio = np.append(users_bio, [user_bio_embed.tolist()], axis=0)
            users_tweets_embed = np.append(
                users_tweets_embed, [user_tweets_embed.tolist()], axis=0)

    logging.info("start pca.")
    bio_pca = PCA(n_components=50)
    tweet_pca = PCA(n_components=100)
    users_bio = bio_pca.fit_transform(users_bio)
    logging.info("users pca finished.")
    users_tweets_embed = tweet_pca.fit_transform(users_tweets_embed)
    logging.info("tweets pca finished.")
    return user_ids, users_bio, users_tweets_embed


def users_feature_exctraction(users_ids, users_bio, users_tweet, my_data):
    data_set = {}
    logging.info("start making feature vectors of users.")
    for i in range(len(users_ids)):
        user = users_ids[i]
        user_bio = users_bio[i].tolist()
        user_tweet = users_tweet[i].tolist()

        profile_background_tile = 1 if my_data[user]['profile_features']['profile_background_tile'] else 0
        profile_use_background_image = 1 if my_data[user][
            'profile_features']['profile_use_background_image'] else 0
        screen_name = len(my_data[user]['profile_features']['screen_name'])
        verified = 1 if my_data[user]['profile_features']['verified'] else 0
        statuses_count = my_data[user]['profile_features']['statuses_count']
        favourites_count = my_data[user]['profile_features']['favourites_count']
        has_extended_profile = 1 if my_data[user]['profile_features']['has_extended_profile'] else 0
        friends_count = my_data[user]['profile_features']['friends_count']
        followers_count = my_data[user]['profile_features']['followers_count']
        number_cascades = len(my_data[user]['cascades_feature'])

        user_feature_vect = [
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
        ] + user_bio + user_tweet

        data_set[user] = user_feature_vect
        logging.info("user {0} added. index = {1}.".format(user, i))
    logging.info('writing output to users_feature.p')
    file_to_write = open('users_feature.p', 'wb')
    pickle.dump(data_set, file_to_write)
    logging.info("finished.")


def users_clustering(users_ids, users_bio, users_tweet, my_data):
    users_dataset = []
    data_set = {}
    for i in range(len(users_ids)):
        user = users_ids[i]
        user_bio = users_bio[i].tolist()
        user_tweet = users_tweet[i].tolist()

        profile_background_tile = 1 if my_data[user]['profile_features']['profile_background_tile'] else 0
        profile_use_background_image = 1 if my_data[user][
            'profile_features']['profile_use_background_image'] else 0
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
    kproto_init = kprototypes.KPrototypes(
        n_clusters=3600, init="Huang", verbose=2, n_init=1)
    logging.info('go for learning clusters')
    result = kproto_init.fit_predict(users_dataset, categorical=[0, 1, 3, 6])
    logging.info("model fit-predict result:{0}".format(result))
    pickle.dump(result, open('results1_text.p', 'wb'))
    pickle.dump(data_set, open('results11_text.p', 'wb'))
    with open('results1_text.txt', 'w') as f:
        f.write("\n".join(str(result)))


def cluster_from_pickle(number_of_clusters=3600):
    user_features = pickle.load(
        open(os.path.join(ROOT_DIR, 'users_feature.p'), 'rb'))

    users_features_vectors = list(user_features.values())
    users_dataset = np.array(users_features_vectors)
    print(users_dataset[1])
    kproto_init = kprototypes.KPrototypes(
        n_clusters=number_of_clusters, init="Huang", verbose=2, n_init=1)
    result = kproto_init.fit_predict(users_dataset, categorical=[0, 1, 3, 6])

    clustering_result = {}
    for i in range(len(result)):
        if result[i] in clustering_result:
            clustering_result[result[i]] += [users_features_vectors[i]]
        else:
            clustering_result[result[i]] = [users_features_vectors[i]]
    file_to_write = open('users_vectprs_clustering.p', 'wb')
    pickle.dump(clustering_result, file_to_write)

    # cluster_vectors = np.array([[0. for i in range(len(users_dataset[0]))] for i in range(number_of_clusters)])
    # for i in range(len(result)):
    # 	cluster_vectors[result[i]] = np.add(cluster_vectors[result[i]], users_dataset[i])
    # return cluster_vectors


def gower_distance(X):
    """
    This function expects a pandas dataframe as input
    The data frame is to contain the features along the columns. Based on these features a
    distance matrix will be returned which will contain the pairwise gower distance between the rows
    All variables of object type will be treated as nominal variables and the others will be treated as 
    numeric variables.
    Distance metrics used for:
    Nominal variables: Dice distance (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    Numeric variables: Manhattan distance normalized by the range of the variable (https://en.wikipedia.org/wiki/Taxicab_geometry)
    """
    X = pd.DataFrame(X)
    individual_variable_distances = []
    print(type(X))
    for i in range(X.shape[1]):
        feature = X.iloc[:, [i]]
        if feature.dtypes[0] == np.object:
            feature_dist = DistanceMetric.get_metric(
                'dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric(
                'manhattan').pairwise(feature) / np.ptp(feature.values)

        individual_variable_distances.append(feature_dist)

    return np.array(individual_variable_distances).mean(0)

def scikit_clustering_ver2(number_of_clusters=3600):
    user_features = pickle.load(
    open(os.path.join(ROOT_DIR, 'users_feature.p'), 'rb'))
    users_features_vectors = list(user_features.values())
    users_dataset = np.array(users_features_vectors)
    df = pd.DataFrame(users_dataset)
    df[0] = df[0].astype('category')
    df[1] = df[1].astype('category')
    df[3] = df[3].astype('category')
    df[6] = df[6].astype('category')
    
    abs_scaler = MaxAbsScaler()
    abs_scaler.fit(df[[2,4,5,7,8,9]])
    df[[2,4,5,7,8,9]] = abs_scaler.transform(df[[2,4,5,7,8,9]])
    
    clustering = KMeans(n_clusters=number_of_clusters, verbose=1).fit(df)

    result = clustering.labels_
    logging.info("result: {0}".format(result))
    clustering_result = {}
    for i in range(len(result)):
        if result[i] in clustering_result:
            clustering_result[result[i]] += [users_features_vectors[i]]
        else:
            clustering_result[result[i]] = [users_features_vectors[i]]
    file_to_write = open('users_vectors_clustering.p', 'wb')
    pickle.dump(clustering_result, file_to_write)


def scikit_clustering(number_of_clusters=3600):
    user_features = pickle.load(
        open(os.path.join(ROOT_DIR, 'users_feature.p'), 'rb'))
    users_features_vectors = list(user_features.values())
    users_dataset = np.array(users_features_vectors)
    df = pd.DataFrame(users_dataset)
    df[0] = df[0].astype('category')
    df[1] = df[1].astype('category')
    df[3] = df[3].astype('category')
    df[6] = df[6].astype('category')
    
    abs_scaler = MaxAbsScaler()
    abs_scaler.fit(df[[2,4,5,7,8,9]])
    df[[2,4,5,7,8,9]] = abs_scaler.transform(df[[2,4,5,7,8,9]])
    print(df.iloc[:,[0]].dtypes[0])
    
    clustering = AgglomerativeClustering(
        n_clusters=number_of_clusters, affinity=gower.gower_matrix, linkage='complete'  ).fit(df)

    result = clustering.labels_
    clustering_result = {}
    for i in range(len(result)):
        if result[i] in clustering_result:
            clustering_result[result[i]] += [users_features_vectors[i]]
        else:
            clustering_result[result[i]] = [users_features_vectors[i]]
    file_to_write = open('users_vectors_clustering.p', 'wb')
    pickle.dump(clustering_result, file_to_write)

if __name__ == '__main__':
   # user_dict = pickle.load(open( os.path.join(ROOT_DIR, 'users_data.p'), "rb")) #give address to users_data.p here.
   # user_ids, users_bio, users_tweet = user_embedding(user_dict)
   # users_feature_exctraction(user_ids, users_bio, users_tweet, user_dict)
    scikit_clustering_ver2()


# user_embedding({'12': {'description':'hi to you', 'cascades_feature':[[12, 1, 'this is a test']]},'13': {'description':'hi to me', 'cascades_feature':[[12, 1, 'this is not a test']]}})
