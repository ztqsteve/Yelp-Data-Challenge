import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

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