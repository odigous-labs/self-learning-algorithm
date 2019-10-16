import time
import sys
import os
import numpy as np
from os.path import join
from datetime import datetime
from scipy import spatial

import cv2

sys.path.append('../../')

import gsom.applications.video_highlights.temporal_features.hoof_data_parser as Parser
from gsom.util import utilities as Utils
from gsom.util import display as Display_Utils

from gsom.params import params as Params
from gsom.core4 import core_controller as Core
from gsom.util import kmeans_cluster_gsom as KMeans_Cluster


def recluster_gsom(converted_feature_vector_dictionary, SF, learning_itr, smoothing_irt, temporal_contexts,
                   forget_threshold, dataset, original_frame_list):
    print('Re-clustering process started\n\n')
    count = 0

    cluster_no_threshold = 2
    no_subclusters = 5

    recluster_arr = []
    final_cluster_out = []
    dynamic_highlights = []

    dynamic_recluster_start = time.time()
    excluded_time = 0

    for element in range(len(converted_feature_vector_dictionary)):

        # Init GSOM Parameters
        gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
                                            temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)

        generalise_params = Params.GeneraliseParameters(gsom_params)

        # convert input data to run gsom
        input_data = {
            0: np.matrix(converted_feature_vector_dictionary[element]['feature_vec'])
        }
        # for i in range(len(converted_feature_vector_dictionary[element]['feature_vec'])):
        #     input_data[i] = np.matrix(converted_feature_vector_dictionary[element]['feature_vec'])

        # Setup the age threshold based on the input vector length
        generalise_params.setup_age_threshold(input_data[0].shape[0])

        recluster_excluded_start = time.time()

        # Mock output location
        output_loc = 'temporal/re-cluster/' + str(count)
        output_loc_images = join(output_loc, 'images/')
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
        if not os.path.exists(output_loc_images):
            os.makedirs(output_loc_images)

        recluster_excluded_end = time.time()
        excluded_time += (recluster_excluded_end - recluster_excluded_start)

        # Process the clustering algorithm
        controller = Core.Controller(generalise_params)
        controller_start = time.time()
        result_dict = controller.run(input_vector_db=input_data,
                                     # return the list/map from here
                                     plot_for_itr=0,
                                     output_loc=output_loc
                                     )
        print('Algorithm for ' + str(count) + ' completed in ', round(time.time() - controller_start, 2), '(s)')
        # saved_name = Utils.Utilities.save_object(result_dict, join(output_loc, 'gsom_nodemap_SF-{}'.format(SF)))

        gsom_nodemap = result_dict[0]['gsom']

        # # Display
        # display = Display_Utils.Display(result_dict[0]['gsom'], None)
        # # display.setup_labels_for_gsom_nodemap(labels, 2, 'Latent Space of {} : SF={}'.format(dataset, SF),
        # #                                       join(output_loc, 'latent_space_' + str(SF) + '_hitvalues'))
        # display.setup_labels_for_gsom_nodemap(converted_feature_vector_dictionary[element]['frame_label'], 3,
        #                                       'Latent Space of {} : SF={}'.format(dataset, SF),
        #                                       join(output_loc, 'latent_space_' + str(SF) + '_labels'))

        print('Completed.')
        count += 1

        recluster_arr.append({
            'gsom': gsom_nodemap,
            'frame_labels': converted_feature_vector_dictionary[element]['frame_label'],
            'feature_vec': converted_feature_vector_dictionary[element]['feature_vec']
        })

        kmeans_som = KMeans_Cluster.KMeansSOM()

        gsom_array = kmeans_som._gsom_to_array(gsom_nodemap)
        gsom_array_length = len(gsom_array)
        if gsom_array_length < no_subclusters:
            no_subclusters = gsom_array_length

        gsom_list, centroids, labels = kmeans_som.cluster_GSOM(gsom_nodemap, no_subclusters)

        farthest_clusters = select_farthest_clusters(
            converted_feature_vector_dictionary[element]['cluster_centroid'],
            centroids,
            cluster_no_threshold
        )

        final_cluster_out.append({
            element: farthest_clusters
        })

        frame_node_list = []
        no_of_nodes = len(gsom_list)
        for x in range(no_of_nodes):
            gsom_node_weights = gsom_list[x]
            for key, node in gsom_nodemap.items():
                if len(node.get_mapped_labels()) > 0:
                    if gsom_node_weights.tolist() == node.recurrent_weights[0].tolist():
                        label_indices = node.get_mapped_labels()
                        frame_labels = []
                        for index in label_indices:
                            frame_labels.append(converted_feature_vector_dictionary[element]['frame_label'][index])
                        frame_node_list.append([key, node, labels[x], frame_labels])
                        break

        for each_item in frame_node_list:
            for frame in each_item[3]:
                if each_item[2] == farthest_clusters:
                    highlight_output = join("temporal/re-cluster/highlights/" + str(element) + "/" + str(each_item[2]) + "/")
                    file_path = join("temporal/re-cluster/highlights/" + str(element) + "/" + str(each_item[2]) + "/",
                                     str(frame) + ".jpg")

                    # return the frames from the dynamic highlights
                    dynamic_highlights.append(str(frame))

                    # if not os.path.exists(highlight_output):
                    #     os.makedirs(highlight_output)
                    # cv2.imwrite(file_path, original_frame_list[int(frame)])

    # print(recluster_arr)
    print(final_cluster_out)
    dynamic_recluster_end = time.time()
    print("Dynamic Feature level 2 reclusterd: " + str(dynamic_recluster_end-dynamic_recluster_start-excluded_time))
    return recluster_arr, dynamic_highlights


def cosine_distance(vec1, vec2):
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    if (mag1 > 0.0 and mag2 > 0.0):
        return spatial.distance.cosine(vec1, vec2)
    else:
        return 0.0


def select_farthest_clusters(main_cluster_centroid, sub_cluster_centroid_list, cluster_no_threshold):
    distances = []
    for centroid in range(len(sub_cluster_centroid_list)):
        distances.append(cosine_distance(main_cluster_centroid, sub_cluster_centroid_list[centroid]))
    print("distances: ", distances)
    sorted_cluster_indices = sorted(range(len(distances)), key=lambda k: distances[k])
    # sorted_cluster_indices = sorted(range(len(distances)), key=lambda k: distances[k])[::-1]
    # commented one to select multiple clusters
    # return sorted_cluster_indices[:cluster_no_threshold]
    return sorted_cluster_indices[0]

