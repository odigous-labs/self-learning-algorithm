import os

import numpy as np
import cv2
import glob
import time

def calc_hist(flow):

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=1)

    q1 = ((0 < ang) & (ang <= 45)).sum()
    q2 = ((45 < ang) & (ang <= 90)).sum()
    q3 = ((90 < ang) & (ang <= 135)).sum()
    q4 = ((135 < ang) & (ang <= 180)).sum()
    q5 = ((180 < ang) & (ang <= 225)).sum()
    q6 = ((225 <= ang) & (ang <= 270)).sum()
    q7 = ((270 < ang) & (ang <= 315)).sum()
    q8 = ((315 < ang) & (ang <= 360)).sum()

    histogram = [q1, q2, q3, q4, q5, q6, q7, q8]
    return (histogram)


def process_video(file_path):
    video_hist = []
    original_frame_list = []
    bins_n = 5
    start_t = time.time()
    cap = cv2.VideoCapture(file_path)
    # cap.open(
    #     "/home/pawara/PycharmProjects/Test/"
    #     "Action/2-self-learning-algorithm-results/self-learning-algorithm-results-colab/gsom/applications"
    #     "/video_highlights/data/3.mp4")

    cap.set(3, 32)
    cap.set(4, 32)
    video_existence = os.path.isfile(file_path)
    print(video_existence)
    ret, prev = cap.read()
    print("File")
    print(prev)
    print()
    original_frame_list.append(prev)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.normalize(src=prevgray, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # prevgray = cv2.normalize(src=prevgray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    iter_counter = 1
    total = 0
    all_frames = 0
    while True:

        ret, img = cap.read()
        all_frames+=1
        if (iter_counter %5 != 0):
            iter_counter += 1
            continue
        total+=1
        if ret:
            original_frame_list.append(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prevgray = gray
        iter_counter+=1
        bins = np.hsplit(flow, bins_n)
        out_bins = []

        for b in bins:
            out_bins.append(np.vsplit(b, bins_n))

        frame_hist = []
        for col in out_bins:
            for block in col:
                frame_hist.extend(calc_hist(block))

        video_hist.append(frame_hist)
        if not ret: break
    end_t = time.time()
    print("1 Dynamic HOOF Generator Extractor: " + str(end_t-start_t))
    print("Total Dynamic Images : "+str(total))
    print("All  Images : " + str(all_frames))
    return video_hist, original_frame_list


if __name__ == '__main__':
    print("Starting ")
    print(cv2.__version__)
    path = '../data/'
    file_name = '3.mp4'
    file_list = glob.glob(path+file_name)

    f= file_list[0]
    print(f)

    start_t = time.time()
    video_existence = os.path.isfile(f)
    print(video_existence)
    video_desc, original_frame_list = process_video(f)
    end_t = time.time()

    print("Time: "+str(end_t-start_t))

    # print(len(video_desc))
    # print(video_desc[27])
    # print(video_desc[26])
    # print(video_desc[25])
    # print(len(original_frame_list))