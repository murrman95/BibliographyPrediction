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
with open("data/training_set.txt", "r") as f:
    file =csv.reader(f, delimiter='\t')
    set_file=list(file)
set= np.array([values[0].split(" ") for values in set_file]).astype(int)


#creates the graph
diG=nx.DiGraph()
#adds the list of papers' IDs
diG.add_nodes_from(ID)
#adds the corresponding links between the paper (training set), links when link_test==1
for ID_source_train,ID_sink_train,link_train in set:
    if link_train==1:
        diG.add_edge(ID_source_train,ID_sink_train)

#check the number of edges
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

    ## Sum up of all features
    degree_features = [diG.in_degree(paper1), diG.out_degree(paper1), diG.in_degree(paper2), diG.out_degree(paper2)]
    heuristic_graph_features = [jaccard_coef, adamic_adar_coef, pref_attachement_coef, common_neig]
    node_info_features = [co_occurence_abstract, same_authors, co_occurence_title, years_diff, same_journal, tfidf_sim] # + [twothree_gram] #

    return node_info_features + heuristic_graph_features + degree_features + triad_features
#########

saved = False
train_features= []
if saved:
    train_features= np.load("./save/kaggle/train_features.npy")
y_train=[]
print("Features construction for Learning...")
step=0
for source,sink,link in set:
    step+=1
    if step%1000==0:    print("Step:",step,"/",len(set))
    if not saved:
        train_features.append(features(source,sink))
    y_train.append(link)
train_features=np.array(train_features)
train_features = preprocessing.scale(train_features)
y_train=np.array(y_train)
if not saved:
    np.save("./save/train_features.npy", train_features)


### For kaggle submission
with open("data/testing_set.txt", "r") as f:
    file =csv.reader(f, delimiter='\t')
    set_file=list(file)
set_test= np.array([values[0].split(" ") for values in set_file]).astype(int)
### than make the changes in the for loops

test_features=[]
if saved:
    test_features=np.load("./save/kaggle/test_features.npy")
y_test=[]
print("Features construction for Testing...")
step=0
for source,sink,link in set_test: ##set_test: ##
    step+=1
    if step%1000==0:    print("Step:",step,"/",len(set_test))
    if not saved:
        test_features.append(features(source,sink))
    y_test.append(link)
test_features=np.array(test_features)
test_features = preprocessing.scale(test_features)
y_test=np.array(y_test)
if not saved:
    np.save("./save/test_features.npy", test_features)


# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=300, algorithm='SAMME.R')
clf = clf.fit(train_features, y_train)
pred = list(clf.predict(test_features))
predictions= zip(range(len(set_test)), pred)

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
with open("predictions.csv","w",newline="") as pred1:
    fieldnames = ['id', 'category']
    csv_out = csv.writer(pred1)
    csv_out.writerow(fieldnames)
    for row in predictions:
        csv_out.writerow(row)

#### scored 0.962876 on 90% 10% split against 0.96630 on kaggle