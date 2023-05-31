from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def clustering(X):
    # Aplicar K-means
    kmeans = KMeans(n_clusters=5, random_state=0) # Define el número de clusters deseado
    kmeans.fit(X)
    # Aplicar PCA para reducir la dimensionalidad a 2
    pca = PCA(n_components=2)
    tensor_pca = pca.fit_transform(X)

    # Obtener las etiquetas de los clusters asignados a cada punto
    cluster_labels = kmeans.labels_
    return tensor_pca,cluster_labels,kmeans