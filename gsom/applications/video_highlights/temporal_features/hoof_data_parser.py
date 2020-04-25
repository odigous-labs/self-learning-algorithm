
import gsom.applications.video_highlights.temporal_features.hoof_generator as histogram
import numpy as np
import glob

class InputParser:
    def get_video_hoof_feature_vector(file_path):
        label_prefix = ""
        label_no = 0
        labels_list = []
        video_hoof, original_frame_list = histogram.process_video(file_path)

        for i in range(len(video_hoof)):
            labels_list.append(label_prefix
                                +
                                str(label_no)
                               )
            label_no += 1

        input_database = {
            0: np.matrix(video_hoof)
        }

        return input_database, labels_list, original_frame_list

if __name__ == '__main__':

    path = 'data/'
    file_name = 'videoplayback.mp4'
    file_list = glob.glob(path+file_name)

    video_file_path = path + file_name
    print(video_file_path)

    input_database, labels, original_frame_list = InputParser.get_video_hoof_feature_vector(video_file_path)
    print(len(input_database))
    print(input_database)
    print(len(labels))
    print(labels)