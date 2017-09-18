import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import graphlab
from sklearn.metrics import mean_squared_error


df = pd.read_csv('data/last_2_years_restaurant_reviews.csv')
business_id = df['business_id'].values
user_id = df['user_id'].values
stars = df['stars'].values

df['count'] = 1
df_user_count = df.groupby('user_id').sum()
df_users = df_user_count[df_user_count['count']>200]

flt1 = df['user_id']==df_users.index[0]
flt2 = df['user_id']==df_users.index[1]
flt3 = df['user_id']==df_users.index[2]
flt4 = df['user_id']==df_users.index[3]
df_cleaned = df[flt1|flt2|flt3|flt4]

df_utility = pd.pivot_table(data=df_cleaned, values='stars',  index='user_id',  
	columns='business_id',  fill_value=0)
utility_mat = df_utility.as_matrix()

# Item-based similarity recommender
item_sim_mat = cosine_similarity(utility_mat.T)
least_to_most_sim_indexes = np.argsort(item_sim_mat, axis=1)

neighborhood_size = 75
neighborhoods = least_to_most_sim_indexes[:, -neighborhood_size:]

# for a random user
user_id = 3

n_users = utility_mat.shape[0]
n_items = utility_mat.shape[1]

items_rated_by_this_user = utility_mat[user_id].nonzero()[0]
# Just initializing so we have somewhere to put rating preds
out = np.zeros(n_items)
for item_to_rate in xrange(n_items):
    relevant_items = np.intersect1d(neighborhoods[item_to_rate],items_rated_by_this_user, assume_unique=True) 
    out[item_to_rate] = np.dot(utility_mat[user_id, relevant_items] ,\
        item_sim_mat[item_to_rate, relevant_items]) / \
        item_sim_mat[item_to_rate, relevant_items].sum()

pred_ratings = np.nan_to_num(out)

n = 10
# Get item indexes sorted by predicted rating
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))

# exclude the items that have been rated by user
unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

unrated_index = unrated_items_by_pred_rating[-n:]

print 'Top 10 restaurants recommended for the user:\n'
for index in unrated_index:
    for name in df[df['business_id']==df_utility.columns[index]]['name']:
        print name
        break

# Matrix Factorization recommender using Graphlab
sf = graphlab.SFrame(df[['user_id', 'business_id', 'stars']])
rec = graphlab.recommender.factorization_recommender.create(
            sf,
            user_id='user_id',
            item_id='business_id',
            target='stars',
            solver='als',
            side_data_factorization=False)

predictions = rec.predict(sf)
rmse = np.sqrt(mean_squared_error(sf['stars'], predictions))

print "graphlab's reported rmse:", rec['training_rmse']
print "calculated rmse:", rmse  



