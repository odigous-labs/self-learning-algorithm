
import time
import sys
import os
from os.path import join
from datetime import datetime

sys.path.append('../../')

import gsom.applications.video_highlights.temporal_features.hoof_data_parser as Parser
from gsom.util import utilities as Utils
from gsom.util import display as Display_Utils

from gsom.params import params as Params
from gsom.core4 import core_controller as Core
from gsom.util import kmeans_cluster_gsom as KMeans_Cluster

def generate_output_config(dataset, SF, forget_threshold, output_save_location, temporal_contexts):

    # Output data config
    output_save_filename = '{}_data_'.format(dataset)
    filename = output_save_filename + str(SF) + '_T_' + str(temporal_contexts) + '_mage_' + str(
        forget_threshold) + 'itr'
    plot_output_name = join(output_save_location, filename)

    # Generate output plot location
    output_loc = plot_output_name
    output_loc_images = join(output_loc, 'images/')
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    if not os.path.exists(output_loc_images):
        os.makedirs(output_loc_images)

    return output_loc, output_loc_images

def hoof_to_gsom(
        SF, learning_itr, smoothing_irt, plot_for_itr, forget_threshold, data_filename, dataset, output_save_location,
        temporal_contexts):

    dynamic_feature_ex_start = time.time()

    # Process the input files
    input_database, labels, original_frame_list = Parser.InputParser.get_video_hoof_feature_vector(data_filename)

    dynamic_feature_ex_end = time.time()

    print("\n video_hoof_gsom Dynamic Feature Extraction Process: " + str(dynamic_feature_ex_end-dynamic_feature_ex_start))

    dynamic_node_first_start = time.time()

    # Init GSOM Parameters
    gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
                                        temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)

    generalise_params = Params.GeneraliseParameters(gsom_params)

    dynamic_node_first_1 = time.time()

    # input_data_features_list, labels = get_video_feature_vector()
    output_loc, output_loc_images = generate_output_config(dataset, SF, forget_threshold, output_save_location, temporal_contexts)

    dynamic_node_first_2 = time.time()

    # Setup the age threshold based on the input vector length
    generalise_params.setup_age_threshold(input_database[0].shape[0])

    # Process the clustering algorithm algorithm
    controller = Core.Controller(generalise_params)
    controller_start = time.time()
    result_dict = controller.run(input_vector_db=input_database,  # return the list/map from here
                                 plot_for_itr=plot_for_itr,
                                 output_loc=output_loc_images
                                 )
    dynamic_node_first_end = time.time()
    print("video_hoof_gsom Dynamic Feature first level GSOM Generated: " + str(dynamic_node_first_end-dynamic_node_first_2+dynamic_node_first_1-dynamic_node_first_start))

    print('Algorithms completed in', round(time.time() - controller_start, 2), '(s)')
    saved_name = Utils.Utilities.save_object(result_dict, join(output_loc, 'gsom_nodemap_SF-{}'.format(SF)))

    gsom_nodemap = result_dict[0]['gsom']

    # Display
    display = Display_Utils.Display(result_dict[0]['gsom'], None)
    # display.setup_labels_for_gsom_nodemap(labels, 2, 'Latent Space of {} : SF={}'.format(dataset, SF),
    #                                       join(output_loc, 'latent_space_' + str(SF) + '_hitvalues'))
    display.setup_labels_for_gsom_nodemap(labels, 3, 'Latent Space of {} : SF={}'.format(dataset, SF),
                                          join(output_loc, 'latent_space_' + str(SF) + '_labels'))

    print('Completed.')
    return gsom_nodemap, original_frame_list, input_database

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

    gsom_nodemap, original_frame_list, input_database = hoof_to_gsom(
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

    kmeans_som = KMeans_Cluster.KMeansSOM()
    gsom_list, centroids, labels = kmeans_som.cluster_GSOM(gsom_nodemap, 4)
    print("gsom list")
    print(gsom_list)
    print("centroids")
    print(centroids)
    print("labels")
    print(labels)