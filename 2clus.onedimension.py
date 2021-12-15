from pandas import DataFrame
import matplotlib.pyplot as plt
#sklearn for apply  for KMeans clustering in python
from sklearn.cluster import KMeans
Data = {'R': [26,36,23,24,34,37,36,28,38,30,62,55,59,41,59,56,52,54,67,40,42,44,37,32,41,45,32,45,51,47],
        'W': [90,45,78,79,45,56,88,33,12,45,56,56,49,67,67,67,34,67,68,59,56,23,12,65,54,58,99,17,84,74]
       }
df = DataFrame(Data,columns=['R','W'])
# kmeans(n_clusters=3).fit(df)
model = KMeans(n_clusters=3).fit(df)
# the center of each cluster (in red) represents the mean of
#  all the observations that belong to that cluster.

#as you may also see, the observations that belong to a given 
# cluster are closer to the center of that cluster, in comparison to the centers of other clusters.
centroids = model.cluster_centers_
print(centroids)
plt.scatter(df['R'], df['W'], c= model.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='green', s=50)
plt.show()
#you see 3 clusters with 3 distinct centroids
#hiiii