
import numpy as np
import cv2
import glob

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
    bins_n = 2

    cap = cv2.VideoCapture(file_path)
    cap.set(3, 32)
    cap.set(4, 32)
    ret, prev = cap.read()
    original_frame_list.append(prev)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.normalize(src=prevgray, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # prevgray = cv2.normalize(src=prevgray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    print(prevgray)
    while True:

        ret, img = cap.read()
        if ret:
            original_frame_list.append(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

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

    return video_hist, original_frame_list


if __name__ == '__main__':

    path = 'data/'
    file_name = 'videoplayback.mp4'
    file_list = glob.glob(path+file_name)

    f=path + file_name
    print(f)

    video_desc, original_frame_list = process_video(f)
    print(len(video_desc))
    print(video_desc[27])
    print(video_desc[26])
    print(video_desc[25])
    print(len(original_frame_list))