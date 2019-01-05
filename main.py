import csv
import nltk
import numpy as np
import networkx as nx
import random
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

"""
    Possible changes
 - Lot of work to do on possible features from node_information.csv
 - Should try with nice random forests and XGBoost
 - Add other graph features

 Tiny details:
 -  Put binary = True in CountVectorizer constuction

"""#
with open("data/node_information.csv", "r") as f:
    file = csv.reader(f)
    node = list(file)

ID = [int(i[0]) for i in node]
year=[int(i[1]) for i in node]
title=[i[2] for i in node]
authors=[i[3] for i in node]
name_journal=[i[4] for i in node]
abstract=[i[5] for i in node]


"""
One_hot vectors on abstract (usefull for co_occurence computations in features construction function)
"""
one_hot = CountVectorizer(stop_words="english")
one_hot_matrix = one_hot.fit_transform(abstract)#.todense()
# one_hot_matrix = one_hot_matrix.toarray()
print(one_hot_matrix.shape)
np.set_printoptions(threshold=np.nan)
print(sum(one_hot_matrix[1]))

"""
One_hot vectors on authors (usefull for co_occurence computations in features construction function)
"""
onehot_authors= CountVectorizer()
onehot_authors_matrix=onehot_authors.fit_transform(authors)
# onehot_authors_matrix = onehot_authors_matrix.toarray()
print(onehot_authors_matrix.shape)
print(onehot_authors.get_feature_names())

"""
One_hot vectors on titles (usefull for co_occurence computations in features construction function)
"""
onehot_titles= CountVectorizer()
onehot_titles_matrix=onehot_titles.fit_transform(title)
# onehot_titles_matrix = onehot_titles_matrix.toarray()
print(onehot_titles_matrix.shape)
print(onehot_titles.get_feature_names())

"""
TF-IDF cosine similarity
"""
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(abstract)


"""
23-gram co-occurence ONLY USE THIS FOR THE FINAL GO. IT IS VERY VERY COMPUTATIONALLY DEMANDING
"""
#one_got_23gram = CountVectorizer(binary = True, stop_words = "english", ngram_range = (2,3))
#one_got_23gram_matrix = one_got_23gram.fit_transform(abstract)


#####co_occurence computation (VERY EXPENSIVE)
# co_occurance_abstract=np.dot(cv_matrix,np.transpose(cv_matrix))
# co_occurance_abstract=np.dot(cv_matrix,cv_matrix.T)
"""
construction of the graph
"""
testtrain=0.9
with open("data/training_set.txt", "r") as f:
    file =csv.reader(f, delimiter='\t')
    set_file=list(file)
set= np.array([values[0].split(" ") for values in set_file]).astype(int)

"""
Cut the set for implementation purpose
"""
number = 10000
to_keep = np.random.choice(range(len(set)), number)
set = [set[i] for i in to_keep]

#creates the graph
diG=nx.DiGraph()
#adds the list of papers' IDs
diG.add_nodes_from(ID)
#adds the corresponding links between the paper (training set), links when link_test==1
##we only keep 90% of the set for testing perpose
for ID_source_train,ID_sink_train,link_train in set[:int(len(set)*testtrain)]: #[:int(len(set)*testtrain)]
    if link_train==1:
        diG.add_edge(ID_source_train,ID_sink_train)
#G.edges() to print all the edges

#check the number of edges
# G.number_of_edges()
G = nx.Graph(diG)
print(diG.nodes)

#########
def features(paper1,paper2):
    """
        outputs the array of the features to input for paper1 and paper 2 comparison
    """
    idx_paper1,idx_paper2=ID.index(paper1),ID.index(paper2)
    # print(abstract[ID.index(str(paper1))])
    # print(abstract[idx_paper1])

    #features from info of the nodes
    co_occurence_abstract=np.dot(one_hot_matrix[idx_paper1],one_hot_matrix[idx_paper2].T).toarray()[0][0]
    same_authors=np.dot(onehot_authors_matrix[idx_paper1],onehot_authors_matrix[idx_paper2].T).toarray()[0][0]
    co_occurence_title=np.dot(onehot_titles_matrix[idx_paper1],onehot_titles_matrix[idx_paper2].T).toarray()[0][0]

    #tfidf cosine similarity
    tf1 = tfidf_matrix[idx_paper1]# in case tfidf mat is so large that it's stored as a sparse matrix
    tf2 = tfidf_matrix[idx_paper2]# in case tfidf mat is so largs that it's stared as a sparse matrix
    tfidf_sim = cosine_similarity(tf1, tf2)[0][0]

    # multiplied_idf = np.dot(tf1,tf2.T)
    # tfidf_max = np.amax(multiplied_idf)

    #VERY COMPUTATIONALLY EXPENSIVE
    #twothree_gram = np.sum(one_got_23gram_matrix[idx_paper1].toarray() * one_got_23gram_matrix[idx_paper2].toarray())

    same_journal = int(name_journal[idx_paper1] == name_journal[idx_paper2])
    try:
        distance=len(nx.shortest_path(G, paper1, paper2))
    except:
        distance=0
    years_diff=int(year[idx_paper1])-int(year[idx_paper2])
    ## features over the graph
    jaccard = nx.jaccard_coefficient(G, [(paper1, paper2)])
    for u, v, p in jaccard:
        jaccard_coef= p
    adamic_adar=nx.adamic_adar_index(G, [(paper1, paper2)])
    for u, v, p in adamic_adar:
        adamic_adar_coef= p
    pref_attachement = nx.preferential_attachment(G, [(paper1, paper2)])
    for u, v, p in pref_attachement:
        pref_attachement_coef= p
    common_neig=len(sorted(nx.common_neighbors(G, paper1, paper2)))

    ## features over the directed graph
    triad_features = [0.0]*8
    for w in sorted(nx.common_neighbors(G, paper1, paper2)):
        if G.has_edge(paper1, w) and G.has_edge(w, paper2):
            triad_features[0]+=1
        if G.has_edge(paper1, w) and G.has_edge(paper2, w):
            triad_features[1]+=1
        if G.has_edge(w, paper1) and G.has_edge(w, paper2):
            triad_features[2] += 1
        if G.has_edge(w, paper1) and G.has_edge(paper2, w):
            triad_features[3] += 1
    for i in range(4, 8):
        if triad_features[i-4]!=0:
            triad_features[i] = triad_features[i-4]/common_neig

    ## Katz similarity
    katz = 0
    beta = 0.005
    path_length = []
    for path in nx.all_simple_paths(G, source=source, target=sink, cutoff=3):
        path_length.append(len(path))
    a = np.array(path_length)
    unique, counts = np.unique(a, return_counts=True)
    dict_katz = dict(zip(unique, counts))
    for length in dict_katz:
        katz += dict_katz[length] * beta ** length * length

    ## Sum up of all features
    degree_features = [diG.in_degree(paper1), diG.out_degree(paper1), diG.in_degree(paper2), diG.out_degree(paper2)]
    heuristic_graph_features = [jaccard_coef, adamic_adar_coef, pref_attachement_coef, common_neig, katz]
    node_info_features = [co_occurence_abstract, same_authors, co_occurence_title, years_diff, same_journal, tfidf_sim] # + [twothree_gram] #

    return node_info_features + heuristic_graph_features + degree_features + triad_features
#########



saved = False
train_features= []
if saved:
    train_features= np.load("./save/train_features.npy")
y_train=[]
print("Features construction for Learning...")
step=0
for source,sink,link in set[:int(len(set)*testtrain)]:
    step+=1
    if step%1000==0:    print("Step:",step,"/",int(len(set)*testtrain))
    if not saved:
        train_features.append(features(source,sink))
    y_train.append(link)
train_features=np.array(train_features)
train_features = preprocessing.scale(train_features)
y_train=np.array(y_train)
if not saved:
    np.save("./save/train_features.npy", train_features)

test_features=[]
if saved:
    test_features=np.load("./save/test_features.npy")
y_test=[]
print("Features construction for Testing...")
step=0
for source,sink,link in set[int(len(set)*testtrain):len(set)]: ##set_test: ##
    step+=1
    if step%1000==0:    print("Step:",step,"/",len(set)-int(len(set)*testtrain))
    if not saved:
        test_features.append(features(source,sink))
    y_test.append(link)
test_features=np.array(test_features)
test_features = preprocessing.scale(test_features)
y_test=np.array(y_test)
if not saved:
    np.save("./save/test_features.npy", test_features)


"""
Features vizualization
"""

print(train_features[:10])

import matplotlib.pyplot as plt
# plt.plot(train_features[:, 6])
# plt.show()

# ### PCA decomposition to vizualize features in 2D
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X = pca.fit_transform(train_features)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(X[:,0], X[:,1], c=y_train)
# ax.set_xlabel('1st dimension')
# ax.set_ylabel('2nd dimension')
# ax.set_title("Vizualization of the PCA decomposition (2D)")
# plt.show()

# ### Vizualize selected features on initial data
# feat = (i,j) #select the features to vizualize
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(train_features[:,feat[0]], train_features[:,feat[1]], c=y_train, alpha=0.8)
# ax.set_xlabel(f'Dimension {feat[0]}')
# ax.set_ylabel(f'Dimension {feat[1]}')
# ax.set_title("Vizualization of the features")
# plt.show()

"""
Model phase (training and testing are in the same paragraphs for one method)
"""

def execute_prediction(classifier, classifier_name):
    classifier.fit(train_features, y_train)
    y_pred = list(classifier.predict(test_features))
    f1 = f1_score(y_test, y_pred)
    print(f"F1 score for {classifier_name}:", f1)

# Prediction rate with SVM
execute_prediction(svm.LinearSVC(), "SVM")

#prediction rate with RF
execute_prediction(RandomForestClassifier(n_estimators=100), "Random Forest")

#prediction using logistic regression
from sklearn.linear_model import LogisticRegression
execute_prediction(LogisticRegression(), "Logistic Regression")

#MLP classsifier (best so far)
from sklearn.neural_network import MLPClassifier
execute_prediction(MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(15, 10), random_state=1), "NN MLP with Adam")

# KNN
from sklearn.neighbors import KNeighborsClassifier
execute_prediction(KNeighborsClassifier(3), "KNN k=3")


# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
execute_prediction(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=300, algorithm='SAMME.R'), "AdaBoost")


#Gaussian processs
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# clf=GaussianProcessClassifier(1.0 * RBF(1.0))
# clf = clf.fit(train_features, y_train)
# pred = list(clf.predict(test_features))
# success_rate=sum(y_test==pred)/len(pred)
# print("Success_rate with Gaussian process:",success_rate)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
execute_prediction(GaussianNB(), "Naive Bayes")

# # different SVMs
# ## doesn't converge
# from sklearn.svm import SVC
# execute_prediction(SVC(kernel="rbf"), "SVM (rbf kernel)")

# xgboost
import xgboost as xgb
execute_prediction(xgb.XGBClassifier(max_depth=2,n_estimaters = 200), "XGBoost")