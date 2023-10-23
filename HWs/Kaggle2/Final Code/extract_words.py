import pandas as pd
import numpy as np
from scipy import sparse
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# import nltk
# nltk.download('stopwords')


#https://github.com/duyet/demo-text-classification/blob/master/classification-with-tfidf-svm.ipynb
#https://github.com/sid-thiru/Text-Classification-with-TFIDF-and-sklearn/blob/master/sklearn_classifiers.py


emoji = [':)', '(;', '(:', ';)', '|:', ':|', '@_@', '):', ':(', '),:', ':,(', ')\';', ':\'(', ':D', ':}', '{:', '}:',
         ':{', ':]', ':[', '[:', ']:', '\\:', '/:', ':/', ':\\', ':p', ';p', '*_*',
         ':*', '*:', '^^', '^_^', '^-^', ':O', 'O:', ':o', 'o:']


def read_train_files(path):
    train_data = pd.read_csv(path + "/test_data.csv")
    train_results = pd.read_csv(path + "/train_results.csv")
    return train_data, train_results


def get_words(string):
    string = str(string)
    string = re.sub('[\n\t]', ' ', string)
    for e in emoji:
        if e in string:
            string = string.replace(e, " {} ".format(e))

    string_list = string.split(" ")

    new_list = []
    for word in string_list:
        if word == '':
            continue
        elif word[:5] == "http:":
            # new_list.append(word)
            continue
        elif word[0] == "@":
            continue
        elif word in emoji:
            new_list.append(word)
        else:
            split_word = re.sub('\'[\']+', '\'', word) #just keep one '
            split_word = re.sub('[^A-Za-z\']+', ' ', split_word)
            split_word = re.sub(r'(.)\1+', r'\1\1', split_word.lower()) #remove repeated letters

            split_word = split_word.split(" ")
            new_list += split_word


    new_list = [i for i in new_list if i != '']
    new_list = normalize(new_list)


    return new_list


def add_to_dictionary(text_list, all_words_dict):
    for word in text_list:
        if word in all_words_dict:
            all_words_dict[word] += 1
        else:
            all_words_dict[word] = 1
    return all_words_dict

def normalize(words): #input: lists of words in each sentence
    """
    Removes stop words and lowercases the text.
    Returns a list of words
    """
    stop_words = stopwords.words('english')
    ps = PorterStemmer()
    return [ps.stem(w.lower()) for w in words if (w.lower() not in stop_words)]


def count_all_words():
    with open("processed strings.json", 'r') as file:
        train_data = json.load(file)
    all_words_dict = {}
    for i in range(len(train_data)):
        print(i)
        # row_text = train_data.iloc[i]['text']
        row_text = train_data[i]
        text_list = row_text.split(" ")
        all_words_dict = add_to_dictionary(text_list, all_words_dict)
    # print(all_words_dict)
    all_words_dict_sorted = dict(sorted(all_words_dict.items(), key=lambda item: item[1], reverse=True))
    with open("processed_words_dict_sorted.json", "w") as write_file:
        json.dump(all_words_dict_sorted, write_file, indent=1)


def remove_few_words(words_number_limit):
    with open("processed_words_dict_sorted.json", 'r') as file: #we should still process test file with this dict
        dictt = json.load(file)
    filter_dict = dict((key, value) for (key, value) in dictt.items() if value > words_number_limit)
    print(len(filter_dict))

    with open("processed strings_test.json", 'r') as file:
        strings = json.load(file)
    new_strings_list = []
    for row in strings:
        s = row.split(" ")
        new_string = [word for word in s if word in filter_dict]
        new_string = " ".join(new_string)
        new_strings_list += [new_string]

    with open("processed strings_remove_few_words_test.json", 'w') as file:
        json.dump(new_strings_list ,file)
    return new_strings_list


def process_all_words(path):
    train_data, train_results = read_train_files(path)
    # train_data = train_data.iloc[:10]
    all_words_list = []
    for i in range(len(train_data)):
        print(i)
        row_text = train_data.iloc[i]['text']
        text_list = get_words(row_text)
        all_words_list.append(" ".join(text_list))

    # save the list
    with open("processed strings_test.json", "w") as write_file:
        json.dump(all_words_list, write_file)
    return all_words_list


def tfidf(data, maxx = 0.6, minn = 0.0001):
    tfidf_vectorizer = TfidfVectorizer(max_df = maxx, min_df = minn)
    tfidf_data = tfidf_vectorizer.fit_transform(data).astype("float16")
    words = tfidf_vectorizer.get_feature_names()
    with open("tfidf_words_stem_removed_words_test.json", "w") as write_file:
        json.dump(words, write_file)

    sparse.save_npz("tfidf_matrix_removed_words_test.npz", tfidf_data)
    return tfidf_data


def BoW(data, maxx = 0.6, minn = 0.0001):
    bow_vectorizer = CountVectorizer(max_df = maxx, min_df = minn)
    bow_data = bow_vectorizer.fit_transform(data)
    words = bow_vectorizer.get_feature_names()
    with open("bow_words_stem_test.json", "w") as write_file:
        json.dump(words, write_file)

    sparse.save_npz("bow_matrix_test.npz", bow_data)
    return bow_data


def test_SVM(x_train, x_test, y_train, y_test, x_real_test):
    SVM = SVC(kernel='rbf', verbose=True, C=50, gamma=0.0045)
    SVMClassifier = SVM.fit(x_train, y_train)
    predictions = SVMClassifier.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    print(a, r, p)
    predictions = SVMClassifier.predict(x_real_test)
    # print(list(predictions))
    # print(predictions.shape)
    with open("prediction.json", "w") as write_file:
        json.dump(list(predictions), write_file)
    return a, p, r


def grid_search(x_train, x_test, y_train, y_test, x_real_test):
    ########DT----hyper parameter search
    from sklearn.model_selection import StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = [

        {'max_depth': [None],
         'min_samples_split': [10, 30, 50, 75, 100, 200],
         'min_samples_leaf': [1, 2, 3, 5],
         'criterion': ['gini', 'entropy'],
         'ccp_alpha': [.0002, .00015, .00001, .0001]}

    ]

    metric = 'f1_macro'

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True)

    grid_search = GridSearchCV(
        DecisionTreeClassifier(), tuned_parameters, scoring=metric, cv=cv_strategy
    )
    # y_train_ = sparse.csr_matrix(np.asarray(y_train))
    # y_train_ = np.asarray(y_train, dtype="float32")

    print("fitting to grid search")
    grid_search.fit(x_train, y_train)
    print('Finished!')

    print("Grid scores on development set:")
    print()
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)

    from sklearn.metrics import f1_score, accuracy_score
    net = grid_search.best_estimator_
    net.fit(x_train, y_train)
    y_train_pred = net.predict(x_train)
    y_valid_pred = net.predict(x_test)
    # y_test = net.predict(X_test_BBOW)

    f1_train = accuracy_score(y_train, y_train_pred)

    f1_valid = accuracy_score(y_test, y_valid_pred)
    # f1_test = f1_score(Y_test, y_test, average='macro')
    print(f1_train, f1_valid)


def test_decision_tree(x_train, x_test, y_train, y_test, x_real_test):
    clf = tree.DecisionTreeClassifier(ccp_alpha= 0.000095, criterion= 'entropy', max_depth= None, min_samples_leaf= 7, min_samples_split= 500)
    #'ccp_alpha': 0.0012, 'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 50
    #mine:
    # {'ccp_alpha': 0.0008, 'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 50} 60-70
    # {'ccp_alpha': 0.002, 'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 2} 40-50
    # {'ccp_alpha': 0.0004, 'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 100} 60-70
    # {'ccp_alpha': 0.0002, 'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 200}
    clf = clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average='weighted')
    r = recall_score(y_test, predictions, average='weighted')
    print(a, p, r)
    predictions = clf.predict(x_real_test)
    #     # print(list(predictions))
    #     # print(predictions.shape)
    # predictions = predictions.astype("int")
    with open("prediction_decisiontree500.json", "w") as write_file:
        json.dump(list(predictions), write_file)


def test_random_forest(x_train, x_test, y_train, y_test, x_real_test):
    clf = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy')
    clf = clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average='weighted')
    r = recall_score(y_test, predictions, average='weighted')
    print(a, p, r)
    predictions = clf.predict(x_real_test)
    with open("prediction_random_forest100_entropy.json", "w") as write_file:
        json.dump(list(predictions), write_file)
    print(clf.get_params())


def save_prediction(file):
    with open(file, "r") as f:
        data = json.load(f)

    new_data = []
    for i in range(len(data)):
        if data[i] == 'neutral':
            res = 1
        elif data[i] == 'negative':
            res = 0
        elif data[i] == 'positive':
            res = 2
        new_data.append(res)
    df = pd.DataFrame(new_data, columns =['target'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'id'})
    df.to_csv("prediction_randomforest50_entropy.csv", index=False)



# remove_few_words(200)
# data = process_all_words("Data")
# with open("processed strings_remove_few_words_test.json", 'r') as file:
#     data = json.load(file)
# tfidf(data)
# BoW(data)
# count_all_words()
#load sparse matrix

# with open("tfidf_words_stem_removed_words_test.json", 'r') as file:
#     data = json.load(file)
#     print(len(data))
training = sparse.load_npz("Processed data/tfidf_matrix_lem_removed_words100.npz").astype("float32")
x_real_test = sparse.load_npz("Processed data/tfidf_matrix_lem_removed_words100_test.npz")
# print(training)


train_results = pd.read_csv("Data/train_results.csv")
train_results = list(train_results['target'])


# new_data_test = []
# for i in range(len(train_results)):
#     if train_results[i] == 'neutral':
#         res = 1
#     elif train_results[i] == 'negative':
#         res = 0
#     elif train_results[i] == 'positive':
#         res = 2
#     new_data_test.append(int(res))

# #
# # #training and test data splits
x_train = training[:900000]
x_test = training[900000:]
y_train = train_results[:900000]
y_test = train_results[900000:]


# grid_search(x_train, x_test, y_train, y_test, x_real_test)
test_decision_tree(x_train, x_test, y_train, y_test, x_real_test)



# #test a classifier
# accuracy, precision, recall = test_SVM(x_train, x_test, y_train, y_test, x_real_test)
# test_random_forest(x_train, x_test, y_train, y_test, x_real_test)
# save_prediction("prediction_random_forest50_entropy.json")
