import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pandas as pd
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
from scipy.spatial import distance
from model import *
from functions import *
from plot import *
from cluster import *
from GPTAPI import openaiAPI



import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

df = get_file_content("papers", 1)

#Eliminar conectores
stop_words = set(stopwords.words('english'))
#Obtener palabras claves

#Transformar palabras en vectores

# CICLO
# CICLO
# CICLO
results = []
for text in df['Text']:
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]

    #Transformar palabras en vectores
    tensor_list = []
    for i in range(0, len(words)):
        tensor = embed_text(words[i], model, tokenizer).mean(1)
        tensor_array = tensor.detach().numpy().astype(np.float64)
        tensor_array = tensor_array.flatten()
        tensor_list.append(tensor_array)

    X = np.array(tensor_list)
    X = np.squeeze(X)

    results.append({
        'words': words,
        'tensor_list': tensor_list,
        'X': X
    })
print('\n\n 1   ----------------------  results definidos\n')

# Procesar cada resultado
for result in results:
    words = result['words']
    tensor_list = result['tensor_list']
    X = result['X']

    # Parte de Gráficar los Clusters
    tensor_reduced, cluster_labels, kmeans= clustering(X=X)

    # Definimos el dataframe
    tensor_dataframe= pd.DataFrame()
    tensor_dataframe["words"] = words
    tensor_dataframe["tensor"] = tensor_list
    tensor_dataframe["reduced_X"] = [tensor_reduced[i,0].round(8) for i in range(0,len(tensor_reduced))]
    tensor_dataframe["reduced_Y"] = [tensor_reduced[i,1].round(8) for i in range(0,len(tensor_reduced))]
    tensor_dataframe["label"] = cluster_labels

    # Mostrar el gráfico
    plt.figure(figsize=(20, 16))
    plt.scatter(tensor_reduced[:, 0], tensor_reduced[:, 1], c=cluster_labels)
    plt.title("Palabras Clusterizadas")
    for i, word in enumerate(words):
        plt.text(tensor_reduced[i, 0], tensor_reduced[i, 1], word, ha='center', va='center', fontsize=8)

    # Permitir que el usuario seleccione dos puntos
    print("Haz click en dos puntos para calcular la correlación lineal.")
    points = plt.ginput(2)

    # Ajustar los límites del gráfico para una mejor visualización
    x_margin = (max(tensor_reduced[:, 0]) - min(tensor_reduced[:, 0])) * 0.2
    y_margin = (max(tensor_reduced[:, 1]) - min(tensor_reduced[:, 1])) * 0.2
    plt.xlim(min(tensor_reduced[:, 0]) - x_margin, max(tensor_reduced[:, 0]) + x_margin)
    plt.ylim(min(tensor_reduced[:, 1]) - y_margin, max(tensor_reduced[:, 1]) + y_margin)

    # Ajustar la separación entre los puntos
    plt.tight_layout()
    plt.close()

    x1, y1 = nearest_points(tensor_reduced, points[0])
    x1 = round(x1, 8)
    y1 = round(y1, 8)
    x2, y2 = nearest_points(tensor_reduced, points[1])
    x2 = round(x2, 8)
    y2 = round(y2, 8)
    X=[x1,x2]
    Y=[y1,y2]

    word_1 = tensor_dataframe.loc[(tensor_dataframe["reduced_X"]==x1) & (tensor_dataframe["reduced_Y"]==y1)]
    word_2 = tensor_dataframe.loc[(tensor_dataframe["reduced_X"]==x2) & (tensor_dataframe["reduced_Y"]==y2)]
    word_1 = word_1["words"].values[0]
    word_2 = word_2["words"].values[0]
    print([word_1,word_2])

    # Identificar los clusters a los que pertenecen los puntos seleccionados
    selected_labels = kmeans.predict([[x1, y1], [x2, y2]])

    # Seleccionar los puntos de los clusters elegidos
    cluster_points = [tensor_reduced[cluster_labels == label] for label in selected_labels]
    x = np.concatenate([points[:, 0] for points in cluster_points])
    y = np.concatenate([points[:, 1] for points in cluster_points])

    # Calcular la correlación lineal entre los dos puntos seleccionados
    corr = np.corrcoef(x, y)[0, 1]
    text = f"La correlación lineal entre los puntos seleccionados es: {corr:.2f}"

    # Mostrar el gráfico con los puntos y la correlación lineal
    plt.figure(figsize=(20, 16))
    plt.scatter(tensor_reduced[:, 0], tensor_reduced[:, 1], c=cluster_labels)
    for i, word in enumerate(words):
        plt.text(tensor_reduced[i, 0], tensor_reduced[i, 1], word, ha='center', va='center', fontsize=8)
    plt.plot([x1, x2], [y1, y2], "ro-", label="Puntos seleccionados")
    plt.title("Selección de puntos y correlación lineal")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, va="top")
    plt.legend()
    plt.show()

#openaiAPI(df["Text"],word_1, word_2)
