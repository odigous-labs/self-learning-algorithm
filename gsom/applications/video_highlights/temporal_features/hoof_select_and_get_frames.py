import os
import time
from os.path import join
from datetime import datetime

import cv2
import video_hoof_gsom as HOOF2GSOM
from util import kmeans_cluster_gsom as KMeans_Cluster

def cluster_gsom_nodes(gsom_nodemap, number_of_clusters):
    kmeans_som = KMeans_Cluster.KMeansSOM()

    gsom_list, centroids, labels = kmeans_som.cluster_GSOM(gsom_nodemap, number_of_clusters)

    return gsom_list, centroids, labels

def write_frame_output(output_path, cluster_folder, img_name, img):
    highlight_output = join(output_path+"/", str(cluster_folder))
    file_path = join(output_path+"/", str(cluster_folder),img_name )
    if not os.path.exists(highlight_output):
        os.makedirs(highlight_output)
    print(highlight_output)
    cv2.imwrite(file_path, img)

def get_gsom_node_array(gsom_list, labels):
    frame_list = []
    no_of_nodes = len(gsom_list)
    for x in range(no_of_nodes):
        gsom_node_weights = gsom_list[x]
        for key, node in gsom_nodemap.items():
            if(len(node.get_mapped_labels())>0):
                if (gsom_node_weights.tolist() == node.recurrent_weights[0].tolist()):
                    frame_list.append([key, node, labels[x], node.get_mapped_labels()])
                    break;
    return frame_list

if __name__ == '__main__':
    # GSOM config
    SF = 0.9
    forget_threshold = 60  # To include forgetting, threshold should be < learning iterations.
    temporal_contexts = 1  # If stationary data - keep this at 1
    learning_itr = 80
    smoothing_irt = 40
    plot_for_itr = 4  # Unused parameter - just for visualization. Keep this as it is.

    # File Config
    dataset = 'video'
    data_filename = "data/1.mp4".replace('\\', '/')


    experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    output_save_location = join('output/', experiment_id)

    gsom_nodemap, original_frame_list = HOOF2GSOM.hoof_to_gsom(
        SF,
        learning_itr,
        smoothing_irt,
        plot_for_itr,
        forget_threshold,
        data_filename,
        dataset,
        output_save_location,
        temporal_contexts
    )

    highlight_output = join(output_save_location+"/", "highlight")
    if not os.path.exists(highlight_output):
        os.makedirs(highlight_output)

    gsom_list, centroids, labels = cluster_gsom_nodes(gsom_nodemap, 10)
    # print("gsom list")
    # print(gsom_list)
    # print("centroids")
    # print(centroids)
    # print("labels")
    # print(labels)
    # print(gsom_list[0])
    frame_node_list = get_gsom_node_array(gsom_list, labels)

    for each_item in frame_node_list:
        for frame in each_item[3]:

            write_frame_output(highlight_output, each_item[2], str(frame)+".jpg", original_frame_list[int(frame)])

    print(len(original_frame_list))

