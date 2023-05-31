<<<<<<< HEAD
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
=======
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def clustering(X):
    # Aplicar K-means
    kmeans = KMeans(n_clusters=2, random_state=0) # Define el número de clusters deseado
    kmeans.fit(X)
    # Aplicar PCA para reducir la dimensionalidad a 2
    pca = PCA(n_components=2)
    tensor_pca = pca.fit_transform(X)

    # Obtener las etiquetas de los clusters asignados a cada punto
    cluster_labels = kmeans.labels_
>>>>>>> 8685f53fe8eb51d89c21829d877f7dc23c7a99dd
    return tensor_pca,cluster_labels,kmeans