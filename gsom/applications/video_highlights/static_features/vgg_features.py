import time
import sys
import os
from os.path import join
from datetime import datetime
sys.path.append('../../../')

from util import utilities as Utils
import data_parser as Parser
from util import display as Display_Utils
from util.kmeans_cluster_gsom import KMeansSOM
from util.FrameSeperator import FrameSeperator
from util import video_to_frames

from params import params as Params
from core4 import core_controller as Core


# GSOM config
SF = 0.83
forget_threshold = 80  # To include forgetting, threshold should be < learning iterations.
temporal_contexts = 1  # If stationary data - keep this at 1
learning_itr = 2
smoothing_irt = 2
plot_for_itr = 4  # Unused parameter - just for visualization. Keep this as it is.

# File Config
experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
output_save_location = join('output/', experiment_id)

path_to_input_video = "path to the input video"
path_to_generated_frames = "./generated_frames/"


def generate_output_config(SF, forget_threshold):

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


if __name__ == '__main__':

        # Init GSOM Parameters
        gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
                                            temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)
        generalise_params = Params.GeneraliseParameters(gsom_params)

        video_to_frames.get_frames(video_input_path = path_to_input_video,frame_output_path=path_to_generated_frames);
        # Process the input files
        input_vector_database, labels = Parser.InputParser.parse_input_frames(path_to_generated_frames)
        output_loc, output_loc_images = generate_output_config( SF, forget_threshold)

        # Setup the age threshold based on the input vector length
        generalise_params.setup_age_threshold(input_vector_database[0].shape[0])
        kmeans_cluster = KMeansSOM()
        # Process the clustering algorithm algorithm
        controller = Core.Controller(generalise_params)
        controller_start = time.time()
        result_dict = controller.run(input_vector_database, plot_for_itr, None, output_loc_images)
        print('Algorithms completed in', round(time.time() - controller_start, 2), '(s)')
        saved_name = Utils.Utilities.save_object(result_dict, join(output_loc, 'gsom_nodemap_SF-{}'.format(SF)))

        gsom_nodemap = result_dict[0]['gsom']
        #clusters = kmeans_cluster.cluster_GSOM(gsom_nodemap,3)

        #frame_seperator = FrameSeperator()

        #labeled_clusteres = frame_seperator.seperate_frames(gsom_nodemap,clusters,labels)

        # Display
        # display = Display_Utils.Display(result_dict[0]['gsom'], None)
        # display.setup_labels_for_gsom_nodemap(labels, 2, 'Latent Space of {} : SF={}',
        #                                       join(output_loc, 'latent_space_' + str(SF) + '_labels'))

        gsom_list, centroids, labels = kmeans_cluster.cluster_GSOM(gsom_nodemap, 4)
        print("gsom list")
        print(gsom_list)
        print("centroids")
        print(centroids)
        print("labels")
        print(labels)
        print('Completed.')
