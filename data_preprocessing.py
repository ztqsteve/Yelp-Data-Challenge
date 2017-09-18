import json
import pandas as pd

file_business, file_review = [
    'yelp_dataset_challenge_round9/yelp_academic_dataset_business.json',
    'yelp_dataset_challenge_round9/yelp_academic_dataset_review.json'
]
with open(file_business) as f:
    df_business = pd.DataFrame(json.loads(line) for line in f)

# Create filtered DataFrame that selects restaurants in Las Vegas
df_business_cate_noNa = df_business.categories.dropna()
df_business_cate_noNa = df_business_cate_noNa.reset_index()

filter_category = list()
for index, line in enumerate(df_business_cate_noNa.categories):
    for word in line:
        if word == "Restaurants":
            filter_category.append(index)
df_business_cate_noNa = df_business_cate_noNa.ix[filter_category]
df_business_cate_noNa = df_business_cate_noNa.set_index('index')
df_filtered1 = pd.merge(df_business_cate_noNa,df_business, how='inner',left_index = True, right_index = True)
df_filtered1 = df_filtered1.drop("categories_x", axis=1)
df_filtered1 = df_filtered1.rename(columns={'categories_y':'categories'})
filter_city = df_filtered1.city == "Las Vegas"
df_filtered = df_filtered1[filter_city]

# Keep selected features
selected_features = [u'business_id', u'name', u'categories', u'stars']
df_selected_business = df_filtered[selected_features]
df_selected_business = df_selected_business.rename(columns={'stars':'avg_stars'})
df_selected_business.to_csv("data/selected_business.csv", index = False, encoding = 'utf-8')

# Use the "business_id" column to filter review data
with open(file_review) as f:
    df_review = pd.DataFrame(json.loads(line) for line in f)

df_left = pd.read_csv("data/selected_business.csv")
df_left = df_left.set_index('business_id')
df_right = df_review.set_index('business_id')
df_join = pd.merge(df_left,df_right,how='inner',left_index = True,right_index = True)
df_join = df_join.reset_index()

#Keep comments from last 2 years
filter_date = pd.to_datetime(df_join.date) > "2015-01-20"
df_final = df_join[filter_date]
df_final.to_csv("data/last_2_years_restaurant_reviews.csv", index = False, encoding = 'utf-8')