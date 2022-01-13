from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from median import pack_colored_points
from sklearn.pipeline import Pipeline
import numpy as np
from itertools import chain
import nltk



def load_text_data(n):
    #X, y = fetch_20newsgroups(return_X_y=True)
    nltk.download("reuters")
    documents=nltk.corpus.reuters.fileids()
    train_docs_id = documents # list(filter(lambda doc: doc.startswith("train"), documents)) 
    X = [nltk.corpus.reuters.raw(doc_id) for doc_id in train_docs_id]
    
    if n < len(X):
        X = np.random.choice(X, n, replace=False)
    #vectorizer = CountVectorizer()
    #vectorizer.fit(X)
    #X_tf = vectorizer.transform(X)
    N_COMPONENTS = 10
    #lsa = TruncatedSVD(n_components=N_COMPONENTS).fit(X_tf)
    transformer = Pipeline(
        [("tf", CountVectorizer()), 
         ("lsa", TruncatedSVD(n_components=N_COMPONENTS)),
         ]).fit(X)
    #tsne = TSNE(n_components=3)
    
    #xlsa = transformer.transform(X)
    
    '''
    #xlsa = lsa.transform(X_tf)
    #pca = PCA(n_components=N_COMPONENTS)
    #xpca = pca.fit_transform(xlsa)
    plt.figure() 
    for i in range(N_COMPONENTS):
        for j in range(N_COMPONENTS):
            if i != j:
                plt.scatter(xlsa[:,i], xlsa[:,j])
    plt.figure()    
    plt.scatter(xlsa[:,0], xlsa[:,1])
    plt.figure()    
    plt.scatter(xpca[:,0], xpca[:,1])
    '''
    
    def transform_and_color(data, color):
        transformed = transformer.transform(data)
        #transformed = tsne.fit_transform(transformed)
        return pack_colored_points(
            transformed, np.full((len(data), 1), color))

    def transform_document(text):
        paragraphs = [x.strip() for x in text.split('\n') if x.strip()]
        if len(paragraphs) == 1:
            p1 = paragraphs
            p2, p3 = [], []
        elif len(paragraphs) == 2:
            p1 = paragraphs[:1]
            p2 = paragraphs[1:]
            p3 = []
        else:
            one_third = len(paragraphs) // 3
            p1 = paragraphs[:one_third]
            p2 = paragraphs[one_third:-one_third]
            p3 = paragraphs[-one_third:]
        return ['\n'.join(x) for x in (p1, p2, p3)] 
        '''
        np.vstack((transform_and_color(p1, 0),
                          transform_and_color(p2, 1),
                          transform_and_color(p3, 2)))      
        '''    
    X_paragraphs = map(chain, zip(*map(transform_document, X)))
    
    X_paragraphs_features = [
        np.expand_dims(transform_and_color(list(data), color), 1) 
        for color, data in enumerate(X_paragraphs)]
    points = np.concatenate(X_paragraphs_features, axis=1)
    
    #points = np.vstack(list(map(transform_document, X)))
    '''
    if n < len(points):
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    '''
    return points

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    np.random.seed(7)
    n = 200000
    result = load_text_data(n)
    
    data = result.reshape((-1, result.shape[-1]))
    X = data[:, :-1]
    Y = data[:, -1]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    model = LGBMClassifier(num_leaves=127)
    model.fit(X_train, Y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print("Train accuracy:", accuracy_score(Y_train, y_pred_train))
    print("Test accuracy:", accuracy_score(Y_test, y_pred_test))
    