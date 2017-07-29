# NLP with Naive Bayes

# Lilbraries imported here
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
# delimiter = '\t' is used for .tsv files
# quoting = 3 ignores double quotes (")
dataset = pd.read_csv("Reviews.tsv", delimiter = '\t', quoting = 3)

# Cleaning the dataset
import re
import nltk
nltk.download('stopwords')		#stopwords helps us remove irrelevant words (e.g. a, the, this, that, etc.)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

listCorpus = []
for i in range(0, 1000):
	# Remove any integers and special characters from reviews, convert to lowercase and split individual words
	review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
	review = reviews.lower()
	review = review.split()

	# Stemming: getting root of words
	# Go through all words in the review, check if they're present in stopwords' english library
	# For large input values in review, use set instead of list for faster performance
	# set(stopwords.words('english'))
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

	# Join back words in the list
	review = ' '.join(review)
	listCorpus.append(review)

# Creating bag of words model using tokenization and sparse matrix
# CountVectorizer allows us to clean the input too, but manual cleaning provides more flexibility
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# max_features gives us the most frequently used words in the dataset
# We can change the number as per requirements

#X is the independent variable, y is dependent
X = cv.fit_transform(listCorpus).toarray()
y = dataset.iloc[:, 1].values
# dataset.iloc[:, 1] gives us 2nd column of the dataset. usually this will have values 0 or 1
# 0 indicating a negative review, 1 indicating positive review

# Splitting dataset into Training and Test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

