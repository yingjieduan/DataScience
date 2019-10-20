import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets.samples_generator import make_blobs
import math as mt
from sklearn import metrics
import utility as util
from scipy.spatial import distance_matrix
from scipy.cluster import hierarchy

class ClusterEvaluation(object):

    evaluation_calinski_harabasz = {}        # Variance Ratio Criterion valuation (the larger the better)
    evaluation_silhouette_score = {}         # Variance Ratio silhouette_score (the larger the better)

    def evaluatte(self, k, X, labels):
        self.evaluation_calinski_harabasz[k] = metrics.calinski_harabasz_score(X, labels)
        self.evaluation_silhouette_score[k] = metrics.silhouette_score(X, labels)

    def showClusterNumberVsEvaluation(self, evaluationType = 'calinski harabasz'):
        """
        :param evaluationType: either 'calinski harabasz' or 'silhouette score'
        :return:
        """
        if evaluationType == 'calinski harabasz':
            plt.plot(list(self.evaluation_calinski_harabasz.keys()), list(self.evaluation_calinski_harabasz.values()))
        else:
            plt.plot(list(self.evaluation_silhouette_score.keys()), list(self.evaluation_silhouette_score.values()))
        plt.show()


class KMeansAutoCluster(ClusterEvaluation):
    k_means_labels = None
    k_means_cluster_centers = None
    best_k = None

    max_k = None

    def run(self, X):

        _X = util.standardize_dataset(X)
        self.max_k = int(mt.log(len(_X))) + 1

        highest_valuation_calinski_harabasz = None

        for k in range(self.max_k):
            if k < 2:
                continue

            n = k * 3  # Number of time the k-means algorithm will be run with different centroid seeds.
            k_means = KMeans(init="k-means++", n_clusters=k, n_init=n)
            k_means.fit(_X)

            super().evaluatte(k, X, k_means.labels_)

            if highest_valuation_calinski_harabasz is None \
                    or highest_valuation_calinski_harabasz < self.evaluation_calinski_harabasz[k]:
                self.best_k = k
                self.k_means_labels = k_means.labels_
                self.k_means_cluster_centers = k_means.cluster_centers_
                highest_valuation_calinski_harabasz = self.evaluation_calinski_harabasz[k]

        return (self.best_k, self.k_means_labels)


class HierarchicalAgglomerativeAutoCluster(ClusterEvaluation):

    labels = None
    k = None       # number of clusters
    _X = None

    def run(self, X):
        self._X = util.standardize_dataset(X)
        self.max_k = int(mt.log(len(self._X))) + 1

        highest_valuation_calinski_harabasz = None

        for k in range(self.max_k):
            if k < 2:
                continue

            n = k * 3  # Number of time the k-means algorithm will be run with different centroid seeds.
            agglom = AgglomerativeClustering(n_clusters = k, linkage = 'complete')
            agglom.fit(self._X)

            super().evaluatte(k, X, agglom.labels_)

            #self.evaluation_calinski_harabasz[k] = metrics.calinski_harabasz_score(_X, agglom.labels_)
            #self.evaluation_silhouette_score[k] = metrics.silhouette_score(_X, agglom.labels_)

            if highest_valuation_calinski_harabasz is None \
                    or highest_valuation_calinski_harabasz < self.evaluation_calinski_harabasz[k]:
                self.best_k = k
                self.labels = agglom.labels_
                highest_valuation_calinski_harabasz = self.evaluation_calinski_harabasz[k]

        return (self.best_k, self.labels)

    def showHierarchicalDiagram(self, orientation = 'top'):
        dist_matrix = distance_matrix(self._X, self._X)
        Z = hierarchy.linkage(dist_matrix, 'complete')
        dendro = hierarchy.dendrogram(Z, labels = self._X, leaf_rotation=0, leaf_font_size =12, orientation = orientation)


class DBSCANAutoCluster(ClusterEvaluation):

    labels = None
    k = None

    def run(self, X, eps=0.5, min_samples=10, metric='euclidean', algorithm='auto', leaf_size=30, n_jobs=None):
        """
        eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN
        parameter to choose appropriately for your data set and distance function.

        min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        This includes the point itself.

        metric : string, or callable
        The metric to use when calculating distance between instances in a feature array. Default euclidean distance.

        algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional, default auto.
        The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest
        neighbors. See NearestNeighbors module documentation for details.
        brute - sparse data
        kd_tree - large data size
        kd_tree - if kd tree training process is quite slow and taking long time.
        auto - usually, auto is good

        leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as
        the memory required to store the tree. The optimal value depends on the nature of the problem.

        n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using
        all processors. See Glossary for more details.
        """
        _X = util.standardize_dataset(X)

        dbacan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size,
                        n_jobs=n_jobs)
        dbacan.fit(_X)

        self.lables = dbacan.labels_
        self.k = len(set(self.lables))

        if self.k < 2:
            return None

        super().evaluatte(self.k, X, self.lables)

        return (self.k, self.lables)


if __name__ == '__main__':
    # generate 2 dimensions data set
    X, y = make_blobs(n_samples=100, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2],
                      random_state=9)

    #plt.scatter(X[:, 0], X[:, 1], marker='.')
    #plt.show()

    module = KMeansAutoCluster()
    best_k, k_means_labels = module.run(X)
    #module.showClusterNumberVsEvaluation()

    module2 = HierarchicalAgglomerativeAutoCluster()
    best_k, labels = module2.run(X)
    #module2.showClusterNumberVsEvaluation()
    module2.showHierarchicalDiagram()

    #plt.plot(list(module2.evaluation_silhouette_score.keys()), list(module2.evaluation_silhouette_score.values()))
    #plt.show()

    module3 = DBSCANAutoCluster()
    k, labels = module3.run(X=X,eps=0.2, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, n_jobs=None)

    plt.scatter(X[:, 0], X[:, 1], marker='.',  c=labels)
    plt.show()
