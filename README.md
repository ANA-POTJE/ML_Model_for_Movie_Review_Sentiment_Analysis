# ML_Model_for_Movie_Review_Sentiment_Analysis

Author: Ana Beatriz Potje

Date: 15/07/2020

ML model to predict positive or negative movie reviews, based on the polarity dataset v2.0 (Pang/Lee ACL 2004)
NLP sentiment analysis project based on the "polarity dataset v2.0" (data source extracted from http://www.cs.cornell.edu/people/pabo/movie-review-data/).
The data set contains 2000 movie review "txt" files (1000 positive and 1000 negative reviews).

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
   - CountVectorizer (Scikit-learn) to count the frequency that words occurs in the text (BAG OF WORDS).
   - TfidfTransformer (Scikit-learn) to re-balance the counts according with size of reviews / number of occurrences (TF-IDF).<br />

The classifier was trained using three different Scikit-learn models, so results could be compared to define the best one:<br />
   - Multinomial Naive Bayes classifier (MultinomialNB from Scikit-learn),<br />
   - Linear Support Vector Classification (LinearSVC from Scikit-learn) and <br />
   - Stochastic Gradient Descent (SGDClassifier from Scikit-learn).<br />
     (Hyperparameters for SGD were tuned using GridSearchCV from Scikit-learn) <br />

Finally, a Pipeline was used in each of the models for Feature extraction and Training steps (Pipeline from Scikit-learn).

# STEP 4: Evaluating the classifier
To evaluate each of the three models we used Accuracy_score and Confusion_matrix metrics, both from the Scikit-learn library.<br />
The results are presented below ... <br />
<br />

# Results:
The model was trained in 80% of the data. The predictions calculated in the TEST SET (remaining 20% of the data) presented the Accuracy Score shown below:

## Multinomial Naive Bayes:

      ACCURACY SCORE 0.805

## Linear Support Vector Classification:

      ACCURACY SCORE 0.825

## Stochastic Gradient Descent (SGD) - BEST MODEL:<br />

      ACCURACY SCORE 0.8275

PS: For SGD it was applied GridSearchCV on the pipeline and the best parameters found were: 
  - clf__alpha: 0.001  <br />
  - clf__penalty: l2  <br />
  - tfidf__use_idf: False  <br />
  - vect__ngram_range: (1, 1) <br />
