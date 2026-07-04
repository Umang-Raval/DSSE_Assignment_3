import os
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.datasets
from gensim.models import CoherenceModel
from gensim import corpora

DOCUMENTS_FOLDER = "txtfiles"

def main():
    # load the data
    files_location = os.path.join(DOCUMENTS_FOLDER)
    documents = sklearn.datasets.load_files(files_location)
    # create topic term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=0.05, stop_words='english', max_features=None)
    X = vectorizer.fit_transform(documents.data)
    # transform data to the gensim
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    vocabulary_gensim = {}
    for key, val in vectorizer.vocabulary_.items():
        vocabulary_gensim[val] = key

    texts = list(vectorizer.vocabulary_.keys())
    id2word = corpora.Dictionary([texts])
    lem = []

    for doc in documents.data:
        l = []
        for word in doc.splitlines():
            l.append(word)
        lem.append(l)
    
    # iterate over min to max topic counts and generate coherence score
    for x in range(3,12):
   
        LDA = gensim.models.LdaMulticore(corpus=corpus_vect_gensim,
                                            id2word=id2word,
                                            num_topics=x, 
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            per_word_topics=True)

        from pprint import pprint # Print the Keyword in the 10 topics
        pprint(LDA.print_topics())        

        score = []
        # calculate coherence
        coherence_model_lda = CoherenceModel(model=LDA, texts=lem,  corpus = corpus_vect_gensim, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda, ' ', x)
        score.append([coherence_lda, x])

if __name__ == '__main__':
    main()