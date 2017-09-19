import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


plt.style.use("ggplot")

df = pd.read_csv('data/last_2_years_restaurant_reviews.csv')

# Define feature variables and target variable
documents = df.text.values
df['perfection'] = df['stars'].apply(lambda x : int(x == 5))
target = df['perfection'].values
print "mean:{}".format(target.mean())

# Create training and test dataset
documents_train, documents_test, target_train, target_test = train_test_split(documents, target, 
	test_size=0.3, random_state=11)

# Get NLP representation of the documents
vectorizer = TfidfVectorizer(stop_words = "english", max_features = 5000)
vectors_train = vectorizer.fit_transform(documents_train).toarray()
words = vectorizer.get_feature_names()
vectors_test = vectorizer.transform(documents_test).toarray()
vectors_all = vectorizer.transform(documents).toarray()

# Use cross validation to evaluate classifiers
lr = LogisticRegression()
score_lr = cross_val_score(lr, vectors_train, target_train, cv=5)
print 'lr cv score:',score_lr

nb = MultinomialNB()
score_nb = cross_val_score(nb, vectors_train, target_train, cv=5)
print 'nb cv score:',score_nb

# choose to tune logistic regression
param_grid = [
  {'C': [0.1,1, 10, 100], 'penalty': ['l1','l2']}
 ]
lr_gridsearch = GridSearchCV(estimator=LogisticRegression(random_state=1),param_grid=param_grid,scoring='roc_auc', cv = 10)
lr_gridsearch.fit(vectors_train, target_train)
print 'best params:',lr_gridsearch.best_params_
print 'best score:',lr_gridsearch.best_score_

lr = LogisticRegression(C=1, penalty='l1')
lr.fit(vectors_train, target_train)
target_test_pred = lr.predict(vectors_test)

def print_results(y_true, y_pred):
    print("Accuracy of the Logistic Regression is: {}".format(accuracy_score(y_true, y_pred)))
    print("Precision of the Logistic Regression is: {}".format(precision_score(y_true, y_pred)))
    print("Recall of the Logistic Regression is: {}".format(recall_score(y_true, y_pred)))
    print("f1-score of the Logistic Regression is: {}".format(f1_score(y_true, y_pred)))
    print("auc score of the Logistic Regression is: {}".format(roc_auc_score(y_true, y_pred)))

print("Test set scores:")
print_results(target_test, target_test_pred)

# ------------Cluster--------------
def choose_k_cluster(train_data,test_data,max_cluster_num):
	'''
	choose number of cluster by silhouette score
	'''
	K=range(2,max_cluster_num+1)
	kmeans = [KMeans(n_clusters=k,max_iter=2000,random_state=113) for k in K]
	[models.fit(train_data) for models in kmeans]
	assigned_cluster = [models.predict(test_data) for models in kmeans]
	s_score = [silhouette_score(test_data, assigned_cluster[i]) for i in range(0,max_cluster_num-1)]
	plt.plot(K,s_score)
    return np.argmax(s_score)+2

n_clusters = choose_k_cluster(vectors_train, vectors_all, 20)
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(vectors_train)

top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
print "top features for each cluster:"
for num, centroid in enumerate(top_centroids):
    print "%d: %s" % (num, ", ".join(words[i] for i in centroid))

#Print out the rating and review of a random sample of the reviews assigned to each cluster to get a sense of the cluster.
assigned_cluster = kmeans.predict(vectors_all)
for i in range(kmeans.n_clusters):
    cluster = np.arange(0, vectors_all.shape[0])[assigned_cluster==i]
    sample_reviews = np.random.choice(cluster, 3, replace=False)
    print "cluster %d:" % i
    for reviews in sample_reviews:
        print 'rate:%d -' % df.ix[reviews]['stars'],
        print "    %s" % df.ix[reviews]['text']

