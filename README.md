# Paso a paso
## Generación de Datos
En este paso, se generan datos ficticios utilizando la función make_blobs de scikit-learn. Se crean 300 puntos de datos distribuidos alrededor de 4 centros. Estos datos podrían representar, por ejemplo, características de pacientes en un estudio médico.
Por ejemplo, Frecuencia cardíaca en reposo, Niveles de colesterol, Índice de masa corporal (IMC) y Hábitos alimenticios y de ejercicio.

_data, labels = make_blobs(n_samples=300, centers=4, random_state=42)_

## Aplicación del Algoritmo de Clustering Jerárquico
Se utiliza el algoritmo de clustering jerárquico para agrupar a los pacientes en clústeres basados en la similitud de sus perfiles de síntomas. 
El método de enlace completo (complete linkage) se utiliza para calcular las distancias entre los clústeres. Esto crea una estructura de árbol que representa cómo se agrupan los datos.

_linked = linkage(data, 'complete')_

## Dibujar el Dendrograma
Se dibuja el dendrograma, que es una representación gráfica de la estructura jerárquica de los clústeres. En el dendrograma, los nodos representan clústeres, y la longitud de las ramas indica la distancia entre los clústeres. Esto proporciona una visión de cómo se agrupan los datos a diferentes niveles de similitud.

_dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrograma de Clustering Jerárquico')
plt.xlabel('Índices de Datos')
plt.ylabel('Distancia')
plt.show()_

## Aplicación del Clustering Jerárquico
Finalmente, se aplica el clustering jerárquico para dividir los datos en un número específico de clústeres (n_clusters). En este caso, se seleccionaron 3 clústeres.
Las etiquetas predichas se asignan a cada punto de datos según su pertenencia al clúster.

_n_clusters = 3
hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete')
labels_pred = hierarchical_cluster.fit_predict(data)_

## Visualización de los Datos Agrupados
Se crea un DataFrame con las características originales y las etiquetas predichas del clúster.
Luego, se visualizan los datos agrupados en un gráfico de dispersión, donde cada color representa un clúster diferente.

_clustered_data = pd.DataFrame(data, columns=['Feature_1', 'Feature_2'])
clustered_data['Cluster'] = labels_pred_

_plt.scatter(clustered_data['Feature_1'], clustered_data['Feature_2'], c=clustered_data['Cluster'], cmap='viridis')
plt.title('Clustering Jerárquico - Datos Agrupados')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()_

