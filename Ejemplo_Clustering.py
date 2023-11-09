import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generar datos ficticios
data, labels = make_blobs(n_samples=300, centers=4, random_state=42)

# Aplicar el algoritmo de clustering jerárquico
# Utilizaremos el método de enlace completo y la métrica de Euclides
linked = linkage(data, 'complete')

# Dibujar el dendrograma
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrograma de Clustering Jerárquico')
plt.xlabel('Índices de Datos')
plt.ylabel('Distancia')
plt.show()

# Aplicar el clustering jerárquico
# Vamos a dividir los datos en 3 clústeres
n_clusters = 3
hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete')
labels_pred = hierarchical_cluster.fit_predict(data)

# Agregar las etiquetas de clúster a los datos
clustered_data = pd.DataFrame(data, columns=['Feature_1', 'Feature_2'])
clustered_data['Cluster'] = labels_pred

# Visualizar los datos agrupados
plt.scatter(clustered_data['Feature_1'], clustered_data['Feature_2'], c=clustered_data['Cluster'], cmap='viridis')
plt.title('Clustering Jerárquico - Datos Agrupados')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
