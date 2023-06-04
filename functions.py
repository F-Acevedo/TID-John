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
import spacy
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from scipy.spatial import distance
import os

def NER(text):
    # Tener instalado Spacy y spacy-transformers
    nlp = spacy.load("en_core_sci_scibert")
    doc = nlp(text)
    # print(list(doc.sents))
    # print("Mostrando entidades")
    words = [ent.text for ent in doc.ents]
    return words

def eliminar_caracteres_extraños(cadena):
    patron = re.compile(r'[^a-zA-Z\s]')
    cadena_limpia = patron.sub('', cadena)
    return cadena_limpia

def embed_text(text, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states

# Obtener los puntos más cercanos a los que el usuario hizo clic
def nearest_points(points, clicked_point):
    points = points.astype(np.float64)  # convierte los puntos a float64
    clicked_point = np.array(clicked_point).astype(np.float64)  # convierte clicked_point a float64
    distances = distance.cdist(points, [clicked_point])
    nearest_index = np.argmin(distances)
    return points[nearest_index]

# Obtener los archivos scrapeados como df
def get_file_content(folder_path, x):
    files = os.listdir(folder_path)
    texts = []
    for i in range(min(x, len(files))):
        file_path = os.path.join(folder_path, files[i])
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        texts.append(content)
    df = pd.DataFrame(texts, columns=['Text'])
    return df