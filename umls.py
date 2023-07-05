import re
import random
import statistics
from functions import NER

def umls(cadena):

# Utilizamos una expresión regular para buscar frases entre corchetes
    patron = r"\[(.*?)\]"

    # Utilizamos la función findall() de re para encontrar todas las coincidencias
    frases = re.findall(patron, cadena)

    # Imprimimos la lista de frases
    for k in range(0,len(frases)):
        frase= NER(frases[k])    
        lista_ratios=[]
        # Generar el ratio para cada par de elementos
        for i in range(len(frase)):
            for j in range(i+1, len(frase)):
                # Generar un número aleatorio entre 0 y 1 exclusivos
                ratio = random.random()
                
                # Agregar el ratio a la lista
                lista_ratios.append(ratio)

        # Calcular el promedio de los ratios
        promedio = statistics.mean(lista_ratios)

        #Imprimir el promedio
        print(f"el promedio de la relación {k+1} es {promedio}")