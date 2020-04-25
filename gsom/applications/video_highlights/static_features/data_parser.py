import sys

sys.path.append('../../')
from gsom.applications.video_highlights.static_features import CreateFeaturesArray


class InputParser:

    @staticmethod
    def parse_input_frames(path_to_frames="./../generated_frames"):
        feature_matrix, labels = CreateFeaturesArray.getFeatureArray(path_to_frames)
        input_database = {
            0: feature_matrix
        }
        return input_database, labels
