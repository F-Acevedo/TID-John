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
import mplcursors
from mpl_interactions import ioff, panhandler, zoom_factory



import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

df = get_file_content("papers", 1)

#Obtener palabras claves

#Transformar palabras en vectores

# CICLO
# CICLO
# CICLO
results = []
for text in df['Text']:
    words=NER(text)
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
    
    def onclick(event):
        if event.button == 3:  # Botón derecho del mouse
            global points
            points.append((event.xdata, event.ydata))
            if len(points) == 2:
                fig.canvas.mpl_disconnect(cid)  # Desconectar el evento de clic después de seleccionar dos puntos
                plt.close()

    # Mostrar el gráfico
    fig, ax = plt.subplots(figsize=(20, 16))
    scatter = ax.scatter(tensor_reduced[:, 0], tensor_reduced[:, 1], c=cluster_labels)
    plt.title("Palabras Clusterizadas")

    for i, word in enumerate(words):
        plt.text(tensor_reduced[i, 0], tensor_reduced[i, 1], word, ha='center', va='center', fontsize=8)

    # Habilitar el zoom con la rueda del mouse
    disconnect_zoom = zoom_factory(ax)
    # Ajustar los límites del gráfico para una mejor visualización
    x_margin = (max(tensor_reduced[:, 0]) - min(tensor_reduced[:, 0])) * 0.2
    y_margin = (max(tensor_reduced[:, 1]) - min(tensor_reduced[:, 1])) * 0.2
    plt.xlim(min(tensor_reduced[:, 0]) - x_margin, max(tensor_reduced[:, 0]) + x_margin)
    plt.ylim(min(tensor_reduced[:, 1]) - y_margin, max(tensor_reduced[:, 1]) + y_margin)

    # Ajustar la separación entre los puntos
    plt.tight_layout()

    # Capturar los puntos seleccionados con el botón derecho del mouse
    points = []
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Mostrar el gráfico interactivo
    plt.show()

    # Obtener las coordenadas seleccionadas
    x1, y1 = nearest_points(tensor_reduced, points[0])
    x1 = round(x1, 8)
    y1 = round(y1, 8)
    x2, y2 = nearest_points(tensor_reduced, points[1])
    x2 = round(x2, 8)
    y2 = round(y2, 8)
    X = [x1, x2]
    Y = [y1, y2]

    word_1 = tensor_dataframe.loc[(tensor_dataframe["reduced_X"] == x1) & (tensor_dataframe["reduced_Y"] == y1)]
    word_2 = tensor_dataframe.loc[(tensor_dataframe["reduced_X"] == x2) & (tensor_dataframe["reduced_Y"] == y2)]
    word_1 = word_1["words"].values[0]
    word_2 = word_2["words"].values[0]
    print([word_1, word_2])



#openaiAPI(df["Text"],word_1, word_2)
