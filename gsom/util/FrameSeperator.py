import collections


class FrameSeperator:

    def __init__(self):
        pass

    def seperate_frames(self, gsom_nodemap, clusters, labels):
        ordered_gsom_nodemap = collections.OrderedDict(gsom_nodemap)
        nodes = list(ordered_gsom_nodemap.values())
        clustered_labels = {}
        for i in range(len(clusters.keys())):
            clustered_labels[i] = []

        for cluster_key, clustered_nodes in clusters.items():
            for node_index in clustered_nodes:
                # clustered_labels[cluster_key].append(gsom_nodemap[gsom_nodemap.keys()[node]].mappedLabels)
                node = nodes[node_index]
                for label_no in node.mappedLabels:
                    clustered_labels[cluster_key].append(labels[label_no])
                pass
        print(clustered_labels)
        return clustered_labels
