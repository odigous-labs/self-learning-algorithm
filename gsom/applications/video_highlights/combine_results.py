import os
from os.path import join
from datetime import datetime
import time

import cv2

import gsom.applications.video_highlights.static_features.vgg_features as StaticHighlights
import gsom.applications.video_highlights.temporal_features.hoof_select_and_get_frames as DynamicHighlights


def write_final_highlights(final_highlights, original_frame_list):
    for frame in final_highlights:
        highlight_output = join("final-highlights/")
        file_path = join("final-highlights/" + str(frame) + ".jpg")

        if not os.path.exists(highlight_output):
            os.makedirs(highlight_output)
        cv2.imwrite(file_path, original_frame_list[int(frame)])


def smooth_highlights(padding, highlights, original_frame_list):
    output = highlights.copy()
    for i in highlights:
        for j in range(1, padding):
            element_pre = int(i) - j
            element_post = int(i) + j
            if not (element_pre < 0):
                if str(element_pre) not in output:
                    output.append(str(element_pre))
            if element_post < len(original_frame_list):
                if str(element_post) not in output:
                    output.append(str(element_post))
    return output


if __name__ == '__main__':
    '''
        Static feature 1st level clustering hyper parameters 
    '''
    static_first_SF = 0.3
    static_first_forget_threshold = 20  # To include forgetting, threshold should be < learning iterations.
    static_first_temporal_contexts = 1  # If stationary data - keep this at 1
    static_first_learning_itr = 100
    static_first_smoothing_itr = 50
    static_first_plot_for_itr = 4

    static_first_experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    static_first_output_save_location = join('static/output/', static_first_experiment_id)

    static_first_dataset = 'video'

    static_first_path_to_input_video = "./data/3.mp4"
    static_first_path_to_generated_frames = "./generated_frames/"

    '''
        Static feature 2st level clustering hyper parameters 
    '''
    static_second_SF = 0.8
    static_second_learning_itr = 100
    static_second_smoothing_itr = 50
    static_second_temporal_contexts = 1
    static_second_forget_threshold = 20
    static_second_dataset = 'video'

    '''
        Dynamic feature 1st level clustering hyper parameters 
    '''
    dynamic_first_SF = 0.3
    dynamic_first_forget_threshold = 20  # To include forgetting, threshold should be < learning iterations.
    dynamic_first_temporal_contexts = 1  # If stationary data - keep this at 1
    dynamic_first_learning_itr = 100
    dynamic_first_smoothing_irt = 50
    dynamic_first_plot_for_itr = 4  # Unused parameter
    # - just for visualization. Keep this as it is.
    dynamic_first_percentage_threshold = 0.01
    # File Config
    dynamic_first_dataset = 'video'
    dynamic_first_video_name = '3.mp4'
    dynamic_first_data_filename = join("./", "data/", dynamic_first_video_name)

    dynamic_first_experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    dynamic_first_output_save_location = join('temporal/output/', dynamic_first_experiment_id)

    '''
        Static feature 2st level clustering hyper parameters 
    '''
    dynamic_second_SF = 0.8
    dynamic_second_learning_itr = 100
    dynamic_second_smoothing_itr = 50
    dynamic_second_temporal_contexts = 1
    dynamic_second_forget_threshold = 20
    dynamic_second_dataset = 'video'

    static_out, static_highlights, original_frame_list = StaticHighlights.run(static_first_SF,
                                                                              static_first_forget_threshold,
                                                                              static_first_temporal_contexts,
                                                                              static_first_learning_itr,
                                                                              static_first_smoothing_itr,
                                                                              static_first_plot_for_itr,
                                                                              static_first_path_to_input_video,
                                                                              static_first_path_to_generated_frames,
                                                                              static_first_output_save_location,
                                                                              static_first_dataset,
                                                                              static_second_SF,
                                                                              static_second_learning_itr,
                                                                              static_second_smoothing_itr,
                                                                              static_second_temporal_contexts,
                                                                              static_second_forget_threshold,
                                                                              static_second_dataset)
    dynamic_out, dynamic_highlights = DynamicHighlights.run(dynamic_first_SF,
                                                            dynamic_first_forget_threshold,
                                                            dynamic_first_temporal_contexts,
                                                            dynamic_first_learning_itr,
                                                            dynamic_first_smoothing_irt,
                                                            dynamic_first_plot_for_itr,
                                                            dynamic_first_data_filename,
                                                            dynamic_first_dataset,
                                                            dynamic_first_output_save_location,
                                                            dynamic_second_SF,
                                                            dynamic_second_learning_itr,
                                                            dynamic_second_smoothing_itr,
                                                            dynamic_second_temporal_contexts,
                                                            dynamic_second_forget_threshold,
                                                            dynamic_second_dataset)

    print('dynamic')
    print(dynamic_highlights)
    print('static')
    print(static_highlights)

    final_highlights = list(set().union(dynamic_highlights, static_highlights))
    print('final highlights')
    print(final_highlights)

    smoothed_highlights = smooth_highlights(2, final_highlights, original_frame_list)
    print('smoothed highlights')
    write_final_highlights(smoothed_highlights, original_frame_list)
    print(smoothed_highlights)
    smoothed_highlights.sort()
    print(smoothed_highlights)
