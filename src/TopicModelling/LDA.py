import sys
import os
from tabnanny import verbose

from sklearn.feature_extraction.text import CountVectorizer
import sklearn.datasets
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from matplotlib import cm
import openpyxl
import math
import numpy as np
from scipy.stats import gaussian_kde

TOPIC_COUNT = 7
TOP_WORDS = 50
DOCUMENTS_FOLDER = "txtfiles"
HTML_FOLDER = "htmlfiles"

def retrieve_int_or_float(input):
    if input.isdigit():
        return int(input)
    else:
        return float(input)

# plot top words of each generated topic
def plot_top_words(model, feature_names, n_top_words, title):
    rows = math.ceil(TOPIC_COUNT / 10)
    columns = math.ceil(TOPIC_COUNT / rows)
    length, height = [columns * 10 , int(TOP_WORDS / 5 * 3)]    
    fig, axes = plt.subplots(1, columns, figsize=(length, height), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx and topic_idx % columns == 0:
            plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
            plt.show()
            fig, axes = plt.subplots(1, columns, figsize=(length, height), sharex=True)
            axes = axes.flatten()
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx % columns]
        ax.barh(top_features, weights, height=0.70) 
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    fig.savefig('topics.png')
    plt.show()

# calculates document to topic distance
#    /\ C
#   /  \
# A/____\ B
def calculate_pos(coords):
    C = (0.5, 0.866) #top of triangle
    A = (0, 0)
    B = (1, 0)
    [a1,b1,c1] = coords
    ac_ratio = c1 / (a1 + c1)
    ab_ratio = b1 / (a1 + b1)
    [x_1, y_1] = [C[0] * ac_ratio, C[1] * ac_ratio] # A_C line dot
    [x_2, y_2] = [ab_ratio, 0] # A_B line dot
    # Y-Y_1 = (Y_2 - Y_1) / (x_2 - x_1) * (X-X_1)
    # [x_1, y_1] [0, 1]
    grad_1 = (0 - y_1) / (1 - x_1)
    c = grad_1 * -x_1 + y_1
    # [0.5, 0.866] [x_2, y_2]
    grad_2 = (y_2 - 0.866) / (x_2 - 0.5)
    d = grad_2 * - 0.5 + 0.866
    #  y = ax + c and y = bx + d 
    # P= (d-c)/(a-b), a* (d-c)/(a-b)+c
    X = (d-c)/(grad_1-grad_2)
    Y = grad_1*X+c
    return [X,Y]

# generates document to topic density graph if topics = 3
def plot_density(values, results):   
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    XY_ = []
    for N in range(values):
        XY_.append(calculate_pos(results[N]))
    
    XY = np.array(XY_)   

    X = XY[:,0]
    Y = XY[:,1]
    xy = np.vstack([X,Y])
    z = gaussian_kde(xy)(xy)
    scatter = ax.scatter(X, Y, c=z)
    fig.suptitle("document topic density. a = 1/topics", fontsize=20)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])
    plt.colorbar(scatter)
    plt.show()

def display_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print( "Topic %d:" % (topic_idx))
        print( " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# retrieve names from the vectorizer and saves them along with their occurences to a file
def get_names(vectorizer, fits):
    names = vectorizer.get_feature_names_out()
    wb = openpyxl.Workbook()
    ws = wb.active
    counts = [sum(x) for x in zip(*fits.toarray())]   
    words_counts = list(zip(names, counts))
    words_counts.sort(key = lambda x: x[1], reverse=True)    
    for wc in words_counts:        
        ws.append([wc[0], wc[1]])
    wb.save("names_r.xlsx")

#used to test if document filenames matches result shape[0] order
def test_fit(vectorizer, lda, results):
    documents = sklearn.datasets.load_files('test_LDA_match/')    
    X = vectorizer.transform(documents.data)
    r = lda.transform(X)
    for n in range(2):
        topic_most_pr = results[n].argmax()        
        print(str(results[n]) + "==" + str(r[n]))
        print(results[n] == r[n])
        for i, val in enumerate(results[n]):
            print(str(results[n][i]) + "==" + str(r[n][i]))
        print("doc: {} topic: {}\n".format(n,topic_most_pr))    
    return

# matches topic number and accuracy with the filenames
def match_name_result(names, values, results):
    wb = openpyxl.Workbook()
    ws = wb.active
    accuracy = []
    for _ in range(TOPIC_COUNT):
        accuracy.append([0.0, 0.0])
    for n in range(values):
        topic_most_pr = results[n].argmax()
        accuracy[topic_most_pr][0] += 1
        accuracy[topic_most_pr][1] += results[n][topic_most_pr]
        topic_most_pr += 1
        ws.append([topic_most_pr, names[n], results[n][topic_most_pr - 1]])        
    wb.save("lda_result_names.xlsx")
    for m in accuracy:
        if (m[0] == 0): # do not divide by zero!
            print(m)
            continue
        print(m[1] / m[0])

def usage():
    print("Use either no args for default or 7 args to set following")
    print("MAX DF int or float")
    print("MIN DF int or float")
    print("MAX Features int or None for default")
    print("ALPHA float")
    print("ETA float")
    print("iterations int")
    print("Amount of topics")
    print("Tau_0 float")


def main():
    global TOPIC_COUNT
    args = sys.argv
    MAX_DF = 0.95
    MIN_DF = 0.05
    ALPHA = None
    ETA = None
    MAX_ITER = 100
    MAX_FEATURES = None    
    TAU_0 = 10.0
    if len (args) != 1 and len(args) != 9:
        usage()
        return
    if len(args) == 9:        
        
        #Vectorizer
        MAX_DF = retrieve_int_or_float(args[1])        
        MIN_DF = retrieve_int_or_float(args[2])
        MAX_FEATURES = None if args[3] == "None" else int(args[3])
        #LDA
        ALPHA = float(args[4])
        ETA = float(args[5])
        MAX_ITER = int(args[6])
        TOPIC_COUNT = int(args[7])
        TAU_0 = float(args[8])   

    # load documents
    files_location = os.path.join(DOCUMENTS_FOLDER)   
    documents = sklearn.datasets.load_files(files_location)
   
    # generate document term matrix
    print("Generating document term matrix.")
    vectorizer = CountVectorizer(max_df=MAX_DF, min_df=MIN_DF, stop_words='english', max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(documents.data)
    
    # lists word occurences in the decreasing order
    get_names(vectorizer, X)

    names = vectorizer.get_feature_names_out()
    
    # perform LDA
    print("Running LDA")
    lda = LatentDirichletAllocation(n_components=TOPIC_COUNT, doc_topic_prior=ALPHA, topic_word_prior=ETA, max_iter=MAX_ITER, 
            learning_method='online', learning_offset= TAU_0, random_state=0, n_jobs = -1, verbose = True)
    results = lda.fit_transform(X)
    
    # list perplexity
    print("perplexity = ", lda.perplexity(results), ". With alpha = ", ALPHA)
    if TOPIC_COUNT == 3:
        plot_density(results.shape[0], results)

    #display_topics(lda, names, TOP_WORDS)
    plot_top_words(lda, names, TOP_WORDS, "Topics in LDA model.")
    #test_fit(vectorizer, lda, results)
    match_name_result(documents.filenames, results.shape[0], results)
    
    
if __name__ == "__main__":
    main()