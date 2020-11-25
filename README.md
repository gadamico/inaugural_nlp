# NLP Bake-Off

Is it possible to identify the political party of U.S. Presidents from their inauguration speeches? That is your challenge in this bake-off!

U.S. Presidents have represented five different political parties in unequal numbers: Democrats, Democratic-Republicans, Federalists, Republicans, and Whigs. Of the 58 inaugural speeches, about 40% were given by representatives of the Democratic Party, and about 40% were given by representatives of the Republican, with the remainder split among the other three parties named above.

In this repo are three pickled `pandas` Series: X_train, y_train, and X_test. The contents of X_train and X_test are the speeches; the contents of y_train are the political parties of the presidents who gave the corresponding speeches in X_train. Your job is to build a model to predict that political parties of the presidents who gave the speeches contained in X_test.

Note that you will need to do some preprocessing of the data. The speeches in X_train (and X_test) are *mostly* clean in the sense of containing little else beyond the English words of the speeches themselves. But you will want to consider strategies like:

- Eliminating capital letters and punctuation;
- Using a stemmer or a lemmatizer;
- Tokenizing; and
- Employing `CountVectorizer()` or `TfidfVectorizer()`

before you get to modeling.

You may use any of the modeling techniques we have explored that apply to classification problems, including any of the modeling classes contained in `sklearn.naive_bayes`.

Note: It would be possible for you to search for the speeches that are contained in X_test and discover who gave them (and also what the speakers' political parties were). For the sake of this bake-off we of course cannot allow that and will consider any such behavior as cheating. You are on your honor!