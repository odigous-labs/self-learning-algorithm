import time
import sys
import os
from os.path import join
from datetime import datetime
import numpy as np
from scipy import spatial
import math

sys.path.append('../../../')

import gsom.applications.video_highlights.static_features.data_parser as Parser
import gsom.applications.video_highlights.static_features.recluster_static as recluster

from gsom.util import utilities as Utils
from gsom.util import display as Display_Utils
from gsom.util.kmeans_cluster_gsom import KMeansSOM
from gsom.util.FrameSeperator import FrameSeperator

from gsom.params import params as Params
from gsom.core4 import core_controller as Core
from gsom.util import vid_2_frame_osh

# GSOM config
# SF = 0.3
# forget_threshold = 80  # To include forgetting, threshold should be < learning iterations.
# temporal_contexts = 1  # If stationary data - keep this at 1
# learning_itr = 2
# smoothing_irt = 2
# plot_for_itr = 4  # Unused parameter - just for visualization. Keep this as it is.

# File Config
# experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
# output_save_location = join('static/output/', experiment_id)
#
# dataset = 'video'
#
# path_to_input_video = "../data/2.mp4"
# path_to_generated_frames = "./generated_frames/"


def generate_output_config(SF, forget_threshold, temporal_contexts, output_save_location):
    # Output data config
    output_save_filename = "output_file"
    filename = output_save_filename + str(SF) + '_T_' + str(temporal_contexts) + '_mage_' + str(
        forget_threshold) + 'itr'
    plot_output_name = join(output_save_location, filename)

    # Generate output plot locationpi
    output_loc = plot_output_name
    output_loc_images = join(output_loc, 'images/')
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    if not os.path.exists(output_loc_images):
        os.makedirs(output_loc_images)

    return output_loc, output_loc_images


def get_gsom_node_array(gsom_nodemap, gsom_list, labels):
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
            if (len(node.get_mapped_labels()) > 0):
                if (gsom_node_weights.tolist() == node.recurrent_weights[0].tolist()):
                    frame_list.append([key, node, labels[x], node.get_mapped_labels()])
                    break
    return frame_list


def get_gsom_node_array_with_new_feature_vectors(gsom_nodemap, gsom_list, labels, input_database, centroids, global_centroid):
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
        print("\nNode:" + str(x))
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
                    break
    return frame_list


def get_gsom_dic_converted_feature_vectors(gsom_nodemap, gsom_list, labels, input_database, global_centroid, cluster_centroids):
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
        gsom_nodemap,
        gsom_list,
        labels,
        input_database,
        cluster_centroids,
        global_centroid
    )

    clusters_dictionary = {}

    for index in range(len(cluster_centroids)):
        clusters_dictionary[index] = {
            'feature_vec': [],
            'frame_label': [],
            'cluster_centroid': cluster_centroids[index],
            'grade_of_frames': []
        }

    for node_frame_list in gsom_node_with_new_feature_vectors:
        clusters_dictionary[node_frame_list[2]]['feature_vec'].extend(node_frame_list[4])
        clusters_dictionary[node_frame_list[2]]['frame_label'].extend(node_frame_list[3])
        clusters_dictionary[node_frame_list[2]]['grade_of_frames'].extend(node_frame_list[5])

    # print(clusters_dictionary)
    return clusters_dictionary


def get_k_and_global_centroid(gsom_nodemap, frame_threshold_for_k):
    gsom_list_middle, global_centroid, labels_middle, k_value = \
        cluster_gsom_nodes_with_selection_K_in_KMeans(gsom_nodemap, frame_threshold_for_k, 1)
    return global_centroid[0], k_value


def cluster_gsom_nodes_with_selection_K_in_KMeans(gsom_nodemap, frame_count_threshold, number_of_clusters):
    kmeans_som = KMeansSOM()
    gsom_list, centroids, labels, k_value = kmeans_som.cluster_GSOM_with_K_selection_in_KMeans(
        gsom_map=gsom_nodemap, element_count_threshold=frame_count_threshold, n_clusters=number_of_clusters
    )
    if (k_value < 2):
        k_value = 2
    return gsom_list, centroids, labels, k_value


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

    return 1


def cosine_distance(vec1, vec2):
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    if (mag1 > 0.0 and mag2 > 0.0):
        return spatial.distance.cosine(vec1, vec2)
    else:
        return 0.0

def run(SF,
        forget_threshold,
        temporal_contexts,
        learning_itr,
        smoothing_irt,
        plot_for_itr,
        path_to_input_video, path_to_generated_frames,
        output_save_location,
        dataset,
        second_level_SF,
        second_level_learning_itr,
        second_level_smoothing_irt,
        second_level_temporal_contexts,
        second_level_forget_threshold,
        second_level_dataset):

    number_of_frames = vid_2_frame_osh.get_frames(video_input_path=path_to_input_video,
                                                     frame_output_path=path_to_generated_frames)
    # Process the input files
    input_vector_database, labels = Parser.InputParser.parse_input_frames(path_to_generated_frames)

    output_loc, output_loc_images = generate_output_config(SF, forget_threshold, temporal_contexts, output_save_location)

    static_node_first_start = time.time()

    # Init GSOM Parameters
    gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
                                        temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)
    generalise_params = Params.GeneraliseParameters(gsom_params)

    # Setup the age threshold based on the input vector length
    generalise_params.setup_age_threshold(input_vector_database[0].shape[0])
    kmeans_cluster = KMeansSOM()
    # Process the clustering algorithm algorithm
    controller = Core.Controller(generalise_params)
    controller_start = time.time()
    result_dict = controller.run(input_vector_database, plot_for_itr, None, output_loc_images)

    static_node_first_end = time.time()
    print("Static Feature first level GSOM Generated: " + str(static_node_first_end - static_node_first_start))

    print('Algorithms completed in', round(time.time() - controller_start, 2), '(s)')
    saved_name = Utils.Utilities.save_object(result_dict, join(output_loc, 'gsom_nodemap_SF-{}'.format(SF)))

    gsom_nodemap = result_dict[0]['gsom']

    static_cluster_first_start = time.time()

    global_centroid, k_value = get_k_and_global_centroid(gsom_nodemap, 10)

    gsom_list, centroids, labels = kmeans_cluster.cluster_GSOM(gsom_nodemap, k_value)

    frame_node_list = get_gsom_node_array(gsom_nodemap, gsom_list, labels)

    static_cluster_first_end = time.time()
    print("Static Feature first level Clustered: " + str(static_cluster_first_end - static_cluster_first_start))

    gsom_dic_converted_feature_vectors = get_gsom_dic_converted_feature_vectors(
        gsom_nodemap,
        gsom_list,
        labels,
        input_vector_database,
        global_centroid,
        centroids
    )

    # print(gsom_dic_converted_feature_vectors)

    """Second level clustering"""
    output, static_highlights = recluster.recluster_gsom(gsom_dic_converted_feature_vectors,
                                                         second_level_SF,
                                                         second_level_learning_itr,
                                                         second_level_smoothing_irt,
                                                         second_level_temporal_contexts,
                                                         second_level_forget_threshold,
                                                         second_level_dataset)

    return output, static_highlights, number_of_frames


if __name__ == '__main__':
    # level 1
    SF = 0.3
    forget_threshold = 20  # To include forgetting, threshold should be < learning iterations.
    temporal_contexts = 1  # If stationary data - keep this at 1
    learning_itr = 100
    smoothing_irt = 50
    plot_for_itr = 4

    experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    output_save_location = join('static/output/', experiment_id)

    dataset = 'video'

    path_to_input_video = "../data/2.mp4"
    path_to_generated_frames = "./generated_frames/"

    # level 2
    second_level_SF = 0.8
    second_level_learning_itr = 100
    second_level_smoothing_irt = 50
    second_level_temporal_contexts = 1
    second_level_forget_threshold = 20
    second_level_dataset = 'video'

    run(SF,
        forget_threshold,
        temporal_contexts,
        learning_itr,
        smoothing_irt,
        plot_for_itr,
        path_to_input_video,
        path_to_generated_frames,
        output_save_location,
        dataset,
        second_level_SF,
        second_level_learning_itr,
        second_level_smoothing_irt,
        second_level_temporal_contexts,
        second_level_forget_threshold,
        second_level_dataset)
    # # Init GSOM Parameters
    # gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
    #                                     temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)
    # generalise_params = Params.GeneraliseParameters(gsom_params)
    #
    # original_frame_list = video_to_frames.get_frames(video_input_path=path_to_input_video, frame_output_path=path_to_generated_frames);
    # # Process the input files
    # input_vector_database, labels = Parser.InputParser.parse_input_frames(path_to_generated_frames)
    # output_loc, output_loc_images = generate_output_config(SF, forget_threshold)
    #
    # # Setup the age threshold based on the input vector length
    # generalise_params.setup_age_threshold(input_vector_database[0].shape[0])
    # kmeans_cluster = KMeansSOM()
    # # Process the clustering algorithm algorithm
    # controller = Core.Controller(generalise_params)
    # controller_start = time.time()
    # result_dict = controller.run(input_vector_database, plot_for_itr, None, output_loc_images)
    # print('Algorithms completed in', round(time.time() - controller_start, 2), '(s)')
    # saved_name = Utils.Utilities.save_object(result_dict, join(output_loc, 'gsom_nodemap_SF-{}'.format(SF)))
    #
    # gsom_nodemap = result_dict[0]['gsom']
    #
    # global_centroid, k_value = get_k_and_global_centroid(gsom_nodemap, 10)
    #
    # gsom_list, centroids, labels = kmeans_cluster.cluster_GSOM(gsom_nodemap, k_value)
    #
    # frame_node_list = get_gsom_node_array(gsom_list, labels)
    #
    # gsom_dic_converted_feature_vectors = get_gsom_dic_converted_feature_vectors(
    #     gsom_list,
    #     labels,
    #     input_vector_database,
    #     global_centroid,
    #     centroids
    # )
    #
    # print(gsom_dic_converted_feature_vectors)
    #
    # """Second level clustering"""
    # output, static_highlights = recluster.recluster_gsom(gsom_dic_converted_feature_vectors,
    #                                                       SF,
    #                                                       learning_itr,
    #                                                       smoothing_irt,
    #                                                       temporal_contexts,
    #                                                       forget_threshold,
    #                                                       dataset,
    #                                                       original_frame_list)
