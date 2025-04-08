##################################
# Plot keypoints on ground truth 
##################################

import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def main():
    parser = argparse.ArgumentParser(description = 'evaluation function of Difference of Gaussian')
    parser.add_argument('--threshold', default = 3.0, type=float, help = 'threshold value for feature selection')
    parser.add_argument('--image_path', default = './testdata/1.png', help = 'path to input image')
    parser.add_argument('--gt_path', default = './testdata/1_gt.npy', help = 'path to ground truth .npy')
    args = parser.parse_args()

    img_ref = cv2.imread(args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float64)

    print(args.threshold)

    DoG = Difference_of_Gaussian(args.threshold)
    keypoints = DoG.get_keypoints(img)

    # read GT
    keypoints_gt = np.load(args.gt_path)

    print(f'GT keypoints = {keypoints_gt.shape[0]}')
    print(f'IM keypoints = {keypoints.shape[0]}')

    # Notice the order in keypoints is (x, y) but (y, x) in imread

    # plot ground truth (blue)
    for pt in keypoints_gt:
        cv2.circle(img_ref, (pt[1], pt[0]), 2, (255, 0, 0), -1)
    
    # plot result (red)
    for pt in keypoints:
        cv2.circle(img_ref, (pt[1], pt[0]), 2, (0, 0, 255), -1)

    cv2.imshow('image', img_ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()