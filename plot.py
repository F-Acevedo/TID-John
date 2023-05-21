import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pandas as pd
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from scipy.spatial import distance



def plot(tensor_pca,cluster_labels,words,X,Y):

    plt.figure(figsize=(8, 6))
    plt.scatter(tensor_pca[:, 0], tensor_pca[:, 1], c=cluster_labels)
    for i, word in enumerate(words):
        plt.text(tensor_pca[i, 0], tensor_pca[i, 1], word)
    plt.plot([X[0], X[1]], [Y[0], Y[1]], "ro-", label="Puntos seleccionados")
    plt.title("Selección de puntos y correlación lineal")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
   # plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, va="top")
    plt.legend()
    plt.show()
