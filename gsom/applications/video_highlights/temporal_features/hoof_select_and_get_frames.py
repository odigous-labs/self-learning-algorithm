import sys
sys.path.append('../../../')
import os
import time
from os.path import join
from datetime import datetime
import numpy as np
import math
from scipy import spatial
import cv2
import video_hoof_gsom as HOOF2GSOM
from util import kmeans_cluster_gsom as KMeans_Cluster

def cluster_gsom_nodes(gsom_nodemap, number_of_clusters):
    kmeans_som = KMeans_Cluster.KMeansSOM()

    gsom_list, centroids, labels = kmeans_som.cluster_GSOM(gsom_nodemap, number_of_clusters)

    return gsom_list, centroids, labels

def cluster_gsom_nodes_with_selection_K_in_KMeans(gsom_nodemap, frame_count_threshold, number_of_clusters ):
    kmeans_som = KMeans_Cluster.KMeansSOM()
    gsom_list, centroids, labels, k_value = kmeans_som.cluster_GSOM_with_K_selection_in_KMeans(
        gsom_map=gsom_nodemap, element_count_threshold=frame_count_threshold, n_clusters=number_of_clusters
    )
    if(k_value<2):
        k_value=2
    return gsom_list, centroids, labels, k_value

def cosine_distance(vec1, vec2):
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    if (mag1 > 0.0 and mag2 > 0.0):
        return spatial.distance.cosine(vec1, vec2)
    else:
        return 0.0

def write_frame_output(output_path, cluster_folder, img_name, img):
    highlight_output = join(output_path+"/", str(cluster_folder))
    file_path = join(output_path+"/", str(cluster_folder),img_name )
    if not os.path.exists(highlight_output):
        os.makedirs(highlight_output)
    print(highlight_output)
    cv2.imwrite(file_path, img)

def get_gsom_node_array(gsom_list, labels):

    """
    :param gsom_list:
    gsom neuron weights returned by the kmeans clustering

    :param labels:
    This is regarding the labels of clusters relevant to each node

    :return:
    For each node combinations of [key, node, labels[x], node.get_mapped_labels()] details will return.
    All details include within a list called frame_list.
    Ex: [[key, node_object, cluster_labels_0 , [frame labels of assigned for the node_object]]
    """

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

def get_gsom_node_array_with_new_feature_vectors(gsom_list, labels, input_database, centroids, global_centroid):
    """
    Advanced Version of the get_gsom_node_array(gsom_list, labels)

    :param gsom_list:
    gsom neuron weights returned by the kmeans clustering

    :param labels:
    This is regarding the labels of clusters relevant to each node

    :param input_database:
    input used for the gsom
    :param centroids:
    all the centroids return from clustering
    :param global_centroid:
    centroid of all nodes in the gsom

    :return:
    similar to the get_gsom_node_array(gsom_list, labels)
    but each first level inner list contains elements as,
        [key, node, labels[x], node.get_mapped_labels(), updated_weights, grade]
    """
    frame_list = []
    no_of_nodes = len(gsom_list)
    print("no of nodes in gsom: " + str(no_of_nodes))

    for x in range(no_of_nodes):
        gsom_node_weights = gsom_list[x]
        print("\nNode:" +str(x))
        for key, node in gsom_nodemap.items():
            if (len(node.get_mapped_labels()) > 0):
                if (gsom_node_weights.tolist() == node.recurrent_weights[0].tolist()):
                    updated_weights = []
                    grade = []
                    for frame in node.get_mapped_labels():

                        prev_feature_vector = input_database[0][int(frame)].tolist()

                        contsant = calculate_const_for_frame(
                            global_centroid,
                            centroids[labels[x]],
                            gsom_node_weights,
                            prev_feature_vector[0]
                        )

                        updated_weights.append(
                            [contsant * val for val in prev_feature_vector[0]]
                        )
                        grade.append(contsant)

                    frame_list.append([key, node, labels[x], node.get_mapped_labels(), updated_weights, grade])
                    break;
    return frame_list

def get_gsom_dic_converted_feature_vectors(gsom_list, labels, input_database, global_centroid, cluster_centroids):
    """This method can use as the starting point for implement second level clustering
    :return:
    return a dictionary with keys as 0, 1, ... total number of clusters
    each cluster has inner level dictionaries as,

        0 : {
            'feature_vec':[vectors for each frame],
            'frame_label':[relevant frame name/no of all feature vector in feature_vec],
            'cluster_centroid':centroid weight vector of each cluster,
            'grade_of_frames':calculated grade for each frame
        }

    """
    gsom_node_with_new_feature_vectors = get_gsom_node_array_with_new_feature_vectors(
        gsom_list,
        labels,
        input_database,
        cluster_centroids,
        global_centroid
    )

    clusters_dictionary = {}

    for index in range(len(cluster_centroids)):
        clusters_dictionary[index] = {
            'feature_vec':[],
            'frame_label':[],
            'cluster_centroid':cluster_centroids[index],
            'grade_of_frames':[]
        }

    for node_frame_list in gsom_node_with_new_feature_vectors:
        clusters_dictionary[node_frame_list[2]]['feature_vec'].extend(node_frame_list[4])
        clusters_dictionary[node_frame_list[2]]['frame_label'].extend(node_frame_list[3])
        clusters_dictionary[node_frame_list[2]]['grade_of_frames'].extend(node_frame_list[5])

    print(clusters_dictionary)
    return clusters_dictionary

def get_num_frame_threshold(total_frames, percentage_threshold):
    return int(total_frames*percentage_threshold)

def calculate_const_for_frame(global_centroid, local_centroid, gsom_node, frame):
    distance_between_global_centroid_cluster_centroid = \
        cosine_distance(global_centroid, local_centroid)
    distance_between_cluster_centroid_node = \
        cosine_distance(local_centroid, gsom_node)
    distance_between_noce_and_datapoint = \
        cosine_distance(gsom_node, frame)

    constant = math.exp((math.exp(distance_between_global_centroid_cluster_centroid) ** 2) * \
                        (math.exp(distance_between_cluster_centroid_node) ** 2) * \
                        (math.exp(distance_between_noce_and_datapoint) ** 2))

    return constant

def get_k_and_global_centroid(gsom_nodemap, frame_threshold_for_k):
    gsom_list_middle, global_centroid, labels_middle, k_value  = \
        cluster_gsom_nodes_with_selection_K_in_KMeans(gsom_nodemap,frame_threshold_for_k, 1)
    return global_centroid[0], k_value

# def main(SF,forget_threshold,temporal_contexts,learning_itr,smoothing_irt,plot_for_itr,
#          percentage_threshold,dataset,video_name,data_filename,experiment_id, output_save_location):



if __name__ == '__main__':
    # GSOM config
    SF = 0.4
    forget_threshold = 60  # To include forgetting, threshold should be < learning iterations.
    temporal_contexts = 1  # If stationary data - keep this at 1
    learning_itr = 100
    smoothing_irt = 50
    plot_for_itr = 4  # Unused parameter
    # - just for visualization. Keep this as it is.
    percentage_threshold = 0.01
    # File Config
    dataset = 'video'
    video_name = '3.mp4'
    data_filename = join("../","data/", video_name)

    experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    output_save_location = join('output/', experiment_id)

    gsom_nodemap, original_frame_list, input_database = HOOF2GSOM.hoof_to_gsom(
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

    highlight_output = join(output_save_location + "/", "highlight")
    if not os.path.exists(highlight_output):
        os.makedirs(highlight_output)

    # num_frame_threshold = get_num_frame_threshold(len(original_frame_list), percentage_threshold)
    # print("num_frame_thgreshold: " + str(num_frame_threshold))

    global_centroid, k_value = get_k_and_global_centroid(gsom_nodemap, 10)

    print("k_value : " + str(k_value))

    gsom_list, centroids, labels = cluster_gsom_nodes(gsom_nodemap, k_value)

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
            write_frame_output(highlight_output, each_item[2], str(frame) + ".jpg", original_frame_list[int(frame)])

    gsom_node_with_new_feature_vectors = get_gsom_node_array_with_new_feature_vectors(
        gsom_list,
        labels,
        input_database,
        centroids,
        global_centroid
    )

    """This method can use as the starting point for implement second level clustering"""
    gsom_dic_converted_feature_vectors = get_gsom_dic_converted_feature_vectors(
        gsom_list,
        labels,
        input_database,
        global_centroid,
        centroids
    )

    print(gsom_dic_converted_feature_vectors[0])
    # print(num_frame_threshold)

    print(len(gsom_dic_converted_feature_vectors[1]['feature_vec']))
    print(len(gsom_dic_converted_feature_vectors[1]['frame_label']))
    print(len(gsom_dic_converted_feature_vectors[1]['cluster_centroid']))
    print(len(gsom_dic_converted_feature_vectors[1]['grade_of_frames']))

    print((gsom_dic_converted_feature_vectors[1]['feature_vec']))
    print((gsom_dic_converted_feature_vectors[1]['frame_label']))
    print((gsom_dic_converted_feature_vectors[1]['cluster_centroid']))
    print((gsom_dic_converted_feature_vectors[1]['grade_of_frames']))

