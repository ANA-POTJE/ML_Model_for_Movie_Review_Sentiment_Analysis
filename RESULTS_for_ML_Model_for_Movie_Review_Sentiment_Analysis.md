The model was trained in 80% of the data.
The predictions calculated in the TEST SET (remaining 20% of the data) presented the Accuracy Score shown below:

1) Multinomial Naive Bayes:

    ACCURACY SCORE 0.805


2) Linear Support Vector Classification 

    ACCURACY SCORE 0.825


3) Stochastic Gradient Descent

    ACCURACY SCORE 0.8275


PS: I applied GridSearchCV on the pipeline and the best parameters found were:
    clf__alpha: 0.001
    clf__penalty: l2
    tfidf__use_idf: False
    vect__ngram_range: (1, 1)


