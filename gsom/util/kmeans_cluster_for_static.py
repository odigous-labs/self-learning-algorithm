from sklearn.cluster import k_means
import collections


class KMeansSOM:

    def _som_to_array(self, som_map):
        som_map_array = []
        for x in range(som_map.shape[0]):
            for y in range(som_map.shape[1]):
                som_map_array.append(som_map[x, y])
        return som_map_array

    def cluster_SOM(self, som_map, n_clusters=2):
        """
        Parameters
        ----------
        som_map : self organizing map
            2D array of weight vectors in SOM.
        n_clusters : number of clusters.

        Returns
        -------
        som_list : list
            list of the som nodes
        centroid : list
            cluster centroids.
        labels : list
            cluster label w.r.t. som node data-point as in som_list
        """

        som_list = self._som_to_array(som_map)

        clf = k_means(som_list, n_clusters=n_clusters)

        centroids = clf[0]
        labels = clf[1]

        return som_list, centroids, labels

    def _gsom_to_array(self, gsom_map):
        gsom_map_array = []
        for key, node in gsom_map.items():
                gsom_map_array.append(node.recurrent_weights[0])
        return gsom_map_array


    def cluster_GSOM(self, gsom_map, n_clusters=2):
        """
        Parameters
        ----------
        gsom_map : growing self organizing map
            2D array of weight vectors in SOM.
        n_clusters : number of clusters.

        Returns
        -------
        gsom_list : list
            list of the gsom nodes
        centroid : list
            cluster centroids.
        labels : list
            cluster label w.r.t. gsom node data-point as in gsom_list
        """

        gsom_list = self._gsom_to_array(gsom_map)
        print("gsom_list")
        print(gsom_list)
        clf = k_means(gsom_list, n_clusters=n_clusters)

        centroids = clf[0]
        labels = clf[1]
        cluster_dict = self.cluster_nodes(labels,gsom_map,n_clusters)

        return cluster_dict


    def cluster_nodes (self,labels,gsom_map,num_of_clusters=2):
        ordered_gsom_nodemap = collections.OrderedDict(gsom_map)
        nodes = list(ordered_gsom_nodemap.values())
        cluster_dict = {}
        for i in range (0,num_of_clusters):
            cluster_dict[i] = []

        for i,label in enumerate(labels):
            cluster_dict[label].append(nodes[i])


        return cluster_dict
