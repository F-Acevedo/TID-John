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

def plot(tensor_pca, cluster_labels, words, X, Y):
    plt.figure(figsize=(20, 16))
    plt.scatter(tensor_pca[:, 0], tensor_pca[:, 1], c=cluster_labels)
    for i, word in enumerate(words):
        plt.text(tensor_pca[i, 0], tensor_pca[i, 1], word, ha='center', va='center', fontsize=8)
    plt.plot([X[0], X[1]], [Y[0], Y[1]], "ro-", label="Puntos seleccionados")
    plt.title("Selección de puntos y correlación lineal")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()

    # Ajustar los límites del gráfico para una mejor visualización
    x_margin = (max(tensor_pca[:, 0]) - min(tensor_pca[:, 0])) * 0.2
    y_margin = (max(tensor_pca[:, 1]) - min(tensor_pca[:, 1])) * 0.2
    plt.xlim(min(tensor_pca[:, 0]) - x_margin, max(tensor_pca[:, 0]) + x_margin)
    plt.ylim(min(tensor_pca[:, 1]) - y_margin, max(tensor_pca[:, 1]) + y_margin)

    # Ajustar la separación entre los puntos
    plt.tight_layout()

    plt.show()