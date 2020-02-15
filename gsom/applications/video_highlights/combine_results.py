import os
from os.path import isfile, join
from datetime import datetime
import time

import cv2
import sys

# sys.path.append(
#     '/content/drive/Shared drives/Final Year Project/7 Implementation Projects/self-learning-algorithm-results/'
# )

sys.path.append('../../../')

# sys.path.append(
#     '/content/drive/Shared drives/Final Year Project/7 Implementation Projects/self-learning-algorithm-results-colab/'
# )

import gsom.applications.video_highlights.static_features.vgg_features as StaticHighlights
import gsom.applications.video_highlights.temporal_features.hoof_select_and_get_frames as DynamicHighlights
# import gsom.util.frames_to_video as FramesToVideo


def write_final_highlights(final_highlights, input_video_path):
    # for frame in final_highlights:
    #     highlight_output = join("final-highlights/")
    #     file_path = join("final-highlights/" + str(frame) + ".jpg")
    #
    #     if not os.path.exists(highlight_output):
    #         os.makedirs(highlight_output)
    #
    #     cv2.imwrite(file_path, original_frame_list[int(frame)])

    if os.path.isfile(input_video_path):

        # Playing the input video from file
        video_capture = cv2.VideoCapture(input_video_path)

        try:
            highlight_output = join("final_highlights/")

            if not os.path.exists(highlight_output):
                os.makedirs(highlight_output)
        except OSError:
            print('Error: Creating directory of data')

        # Capture the very first frame
        return_status, frame = video_capture.read()

        current_frame = 0

        while return_status:
            if current_frame in final_highlights:
                # Saving the current frame's image as a jpg file
                file_path = join("final_highlights/" + str(current_frame) + ".jpg")

                # print("Creating..." + file_path)
                cv2.imwrite(file_path, frame)
                # Increasing the current frame value for the next frame

            current_frame += 1
            # Capture frame-by-frame
            return_status, frame = video_capture.read()
        # Release the capture
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid input video to capture. Location or the video not exist.")


def smooth_highlights(padding, highlights, number_of_frames):
    for i in range(len(highlights)):
        highlights[i] = int(highlights[i]) * 5
    output = highlights.copy()
    for i in highlights:
        for j in range(1, padding):
            element_pre = int(i) - j
            element_post = int(i) + j
            if not (element_pre < 0):
                if element_pre not in output:
                    output.append(element_pre)
            if element_post < number_of_frames:
                if element_post not in output:
                    output.append(element_post)
    return output


def convert_frames_to_video(input_frames_path, output_video_path, output_video_name, fps):
    if os.path.isdir(input_frames_path):

        frame_array = []

        files = [file for file in os.listdir(input_frames_path) if isfile(join(input_frames_path, file))]

        # Sort the frames order by the name
        files = sorted(files, key=lambda x: int(x[:-4]))
        print(files)

        if len(files) != 0:

            try:
                # validate the exitence of the output location to save the output video
                if not os.path.exists(output_video_path):
                    os.makedirs(output_video_path)
            except OSError:
                print('Error: Creating directory of data')

            for i in range(len(files)):
                # reading image files
                filename = input_frames_path + files[i]
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)
                # print(filename)

                # inserting the current frame into the frame array
                frame_array.append(img)

            # Initiate the output video file
            out = cv2.VideoWriter(output_video_path + output_video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for i in range(len(frame_array)):
                # writing to a image array
                out.write(frame_array[i])
            out.release()
        else:
            print("Need to include frames in the required format within the given location of frames.")
    else:
        print("Given path of the frames not exists.")


if __name__ == '__main__':
    '''
        Static feature 1st level clustering hyper parameters 
    '''
    static_first_SF = 0.25
    static_first_forget_threshold = 5  # To include forgetting, threshold should be < learning iterations.
    static_first_temporal_contexts = 1  # If stationary data - keep this at 1
    static_first_learning_itr = 50
    static_first_smoothing_itr = 10
    static_first_plot_for_itr = 4

    static_first_experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    static_first_output_save_location = join('static/output/', static_first_experiment_id)

    static_first_dataset = 'video'

    static_first_path_to_input_video = "./data/3.mp4"
    # static_first_path_to_input_video = './data/motor_bike.mp4'
    static_first_path_to_generated_frames = "./generated_frames/"

    '''
        Static feature 2st level clustering hyper parameters 
    '''
    static_second_SF = 0.75
    static_second_learning_itr = 50
    static_second_smoothing_itr = 10
    static_second_temporal_contexts = 1
    static_second_forget_threshold = 5
    static_second_dataset = 'video'

    '''
        Dynamic feature 1st level clustering hyper parameters 
    '''
    dynamic_first_SF = 0.15
    dynamic_first_forget_threshold = 5  # To include forgetting, threshold should be < learning iterations.
    dynamic_first_temporal_contexts = 1  # If stationary data - keep this at 1
    dynamic_first_learning_itr = 50
    dynamic_first_smoothing_irt = 10
    dynamic_first_plot_for_itr = 4  # Unused parameter
    # - just for visualization. Keep this as it is.
    dynamic_first_percentage_threshold = 0.01
    # File Config
    dynamic_first_dataset = 'video'
    dynamic_first_video_name = '3.mp4'
    # dynamic_first_video_name = 'motor_bike.mp4'
    dynamic_first_data_filename = join("./", "data/", dynamic_first_video_name)

    dynamic_first_experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    dynamic_first_output_save_location = join('temporal/output/', dynamic_first_experiment_id)

    '''
        Static feature 2nd level clustering hyper parameters 
    '''
    dynamic_second_SF = 0.65
    dynamic_second_learning_itr = 50
    dynamic_second_smoothing_itr = 10
    dynamic_second_temporal_contexts = 1
    dynamic_second_forget_threshold = 5
    dynamic_second_dataset = 'video'

    highlight_frames = './final_highlights/'
    video_output_path = './highlight_video/'
    video_name = 'highlights.avi'
    fps_outvid = 24

    static_out, static_highlights, number_of_frames = StaticHighlights.run(static_first_SF,
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

    print('number of frames')
    print(number_of_frames)
    print('dynamic')
    print(dynamic_highlights)
    print(len(dynamic_highlights))
    print('static')
    print(static_highlights)
    print(len(static_highlights))
    final_highlights = list(set().union(dynamic_highlights, static_highlights))
    final_highlights = [int(i) for i in final_highlights]
    print('final highlights')
    print(final_highlights)
    print(len(final_highlights))
    smoothed_highlights = smooth_highlights(2, final_highlights, number_of_frames)
    print('smoothed highlights')
    print(smoothed_highlights)
    smoothed_highlights = sorted(smoothed_highlights)
    print(smoothed_highlights)
    write_final_highlights(smoothed_highlights, static_first_path_to_input_video)
    print(len(smoothed_highlights))
    convert_frames_to_video(highlight_frames, video_output_path, video_name, fps_outvid)
