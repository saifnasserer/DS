from sklearn.cluster import KMeans

def apply_clustering(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    return labels