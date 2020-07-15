# ML_Model_for_Movie_Review_Sentiment_Analysis
ML model to predict positive or negative movie reviews, based on the polarity dataset v2.0 (Pang/Lee ACL 2004)
NLP sentiment analysis project based on the "polarity dataset v2.0" (data source extracted from http://www.cs.cornell.edu/people/pabo/movie-review-data/).
The data set contains 2000 movie review "txt" files (1000 positive and 1000 negative reviews).

# Author: Ana Beatriz Potje
# Date: 15/07/2020

review_polarity.tar.GZ : dataset used for model building.
Movie_Review_2000_files.ipynb: the jupyter notebook containing code.

This project is organized as follows:

# STEP 1: Loading the data
The 2000 text files were loaded in a Python DICT object. 
The data was then split into TRAIN (80%) and TEST (20%)

# STEP 2: Preprocessing the data
Created a customized function called "spacy_tokenizer" using the SPACY library. This function convert the text in TOKENS, then turn the text into LOWERCASE, LEMMATIZE it, and finally remove STOPWORDS and PUNCTUATIONS. We also created two TARGET vectors for the TRAIN and TEST data sets, labelling the reviews as "0" for negative or "1" for positive.

# STEP 3: Extracting the features & Training a classifier
For the feature extraction we used:
   - CountVectorizer was used to count the frequency that words occurs in the text (BAG OF WORDS).
   - TfidfTransformer was used to re-balance the counts according with size of reviews / number of occurrences (TF-IDF).
Finally the model was trained using MULTINOMIAL NAIVE BAYES classifier (MultinomialNB).
A PIPELINE was used for Feature extraction and Training steps.

# STEP 4: Evaluating the classifier
The SKLEARN library was used for evaluating the model (accuracy_score and confusion_matrix metrics).

PS1: The other following models were used in STEP 3:
     Linear Support Vector Classification and 
     Stochastic Gradient Descent (SGD)

PS2: HYPERPARAMETERS for SGD were tuned using GridSearchCV
