# KAPPA BACKEND
from flask import Flask, request, jsonify
from flask_cors import CORS
from load_data import merge_user_rating, load_rating_data
from cluster import kmeans_clustering, dbscan_clustering
from predict import find_neighbor, predict
import pandas as pd
import numpy as np
import timeit

app = Flask(__name__)
CORS(app)

# Rating data
all_data = pd.read_csv('rating_5_min_75.csv')
# Comic genre
comic_genre = pd.read_csv('comic_genre.csv')
# Comic data (images and title)
comic_data = pd.read_csv('firestore_comics.csv')

def find_nearest_unrated(user_input, ratings_cluster):
    rated_comics = [user_input[i]['id'] for i in range(len(user_input))]
    rated_comics = np.array(rated_comics)

    item_to_predict = []

    for item in rated_comics:
        which_cluster = ratings_cluster.loc[ratings_cluster.index ==
                                            item, 'cluster'].iloc[0]
        clustered_ratings = ratings_cluster.loc[ratings_cluster['cluster']
                                                == which_cluster]

        item_to_predict.append(find_neighbor(item, clustered_ratings, usage="nearest_unrated", verbose=False))

    item_to_predict = np.array(item_to_predict).flatten() # convert the array into 1D
    item_to_predict = np.unique(item_to_predict) # remove duplicate item
    item_to_predict = np.setdiff1d(item_to_predict, rated_comics) # remove rated_comics

    return item_to_predict

@app.route('/', methods=['GET'])
def index():
    return "WELCOME TO KAPPA"

@app.route('/api/kmeans', methods=['POST'])
def kmeans():
    # Receive the user input
    user_input = request.get_json()
    print('User input received...')
    # Initialize an empty prediction list
    prediction_list = []

    start = timeit.default_timer()
    print('Processing Data...')
    # Merge user input with all_data
    rating_data = merge_user_rating(user_input, all_data)
    # Load the data for clustering and prediction
    rating_data, cluster_data = load_rating_data(comic_genre, rating_data)
    # Cluster the data
    ratings_cluster, cluster_centroids = kmeans_clustering(2, cluster_data, rating_data)
    # Find nearest unrated item to predict
    item_to_predict = find_nearest_unrated(user_input, ratings_cluster)
    # Predict each item ratings
    for item in item_to_predict:
        print('Predicting {}'.format(item))
        prediction = predict("KAPPA_NEW_USER", item, ratings_cluster, cluster_centroids, verbose=False) 
        title = comic_data.loc[comic_data['comic_id'] == item].iat[0, 1]
        image_url = comic_data.loc[comic_data['comic_id'] == item].iat[0, 2]
        prediction['title'] = title
        prediction['image_url'] = image_url
        prediction_list.append(prediction)
    # Sort the rating by descending order
    prediction_list = sorted(prediction_list, key = lambda i: i['rating'], reverse = True)
    end = timeit.default_timer()
    time = end-start
    print('\nPrediction Result:\n{}'.format(prediction_list))
    print('Time Elapsed: {}'.format(time))

    return jsonify(prediction_list)


@app.route('/api/dbscan', methods=['POST'])
def dbscan():
    # Receive the user input
    user_input = request.get_json()
    print('User input received...')
    # Initialize an empty prediction list
    prediction_list = []

    start = timeit.default_timer()
    print('Processing Data...')
    # Merge user input with all_data
    rating_data = merge_user_rating(user_input, all_data)
    # Load the data for clustering and prediction
    rating_data, cluster_data = load_rating_data(comic_genre, rating_data)
    # Cluster the data
    ratings_cluster = dbscan_clustering(7.8, cluster_data, rating_data)
    # Find nearest unrated item to predict
    item_to_predict = find_nearest_unrated(user_input, ratings_cluster)
    # Predict each item ratings
    for item in item_to_predict:
        print('Predicting {}'.format(item))
        prediction = predict("KAPPA_NEW_USER", item, ratings_cluster, centroids=None, verbose=False) 
        title = comic_data.loc[comic_data['comic_id'] == item].iat[0, 1]
        image_url = comic_data.loc[comic_data['comic_id'] == item].iat[0, 2]
        prediction['title'] = title
        prediction['image_url'] = image_url
        prediction_list.append(prediction)
    # Sort the rating by descending order
    prediction_list = sorted(prediction_list, key = lambda i: i['rating'], reverse = True)
    end = timeit.default_timer()
    time = end-start
    print('\nPrediction Result:\n{}'.format(prediction_list))
    print('Time Elapsed: {}'.format(time))

    return jsonify(prediction_list)

app.run(debug=True)