import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
np.random.seed(999)


# transform keypoints with homography
def xform_homography(H, points):

    one = np.ones((points.shape[0], 1))
    new_points = np.concatenate((points, one), axis=1).T
    ret = np.dot(H, new_points)
    ret = ret/ret[-1,:]

    return ret

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # find matches
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)

        # print(f'\nmatches = {len(matches)}')

        # img3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # cv2.imshow('Image', img3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # TODO: 2. apply RANSAC to choose best H
        best_H = np.eye(3)

        # find match index in kp1, kp2
        kp1_matches = np.array([kp1[i.queryIdx].pt for i in matches])
        kp2_matches = np.array([kp2[i.trainIdx].pt for i in matches])

        # RANSAC parameter
        p = 0.99 # not outlier prob.
        v = 0.5  # inlier prob.
        samples = 5    # sampled point size

        threshold = 0.1
        max_inlier = 0

        NUM_ITERS = int(np.log(1-p)/np.log(1-(1-v)**samples))

        num_matches = len(kp1_matches)

        kp1_px = np.concatenate((kp1_matches.T, np.ones(shape=(1, len(kp1_matches)))), axis=0)

        for i in range(NUM_ITERS):
            sel_idx = np.random.choice(num_matches, samples, replace=False)

            h_i = solve_homography(kp2_matches[sel_idx], kp1_matches[sel_idx])
            new_kp2_px = xform_homography(h_i, kp2_matches)

            dist = np.linalg.norm((new_kp2_px - kp1_px), axis=0)
            dist = dist / len(dist)
            
            inlier_cnt = np.sum(dist < threshold)
            if (inlier_cnt > max_inlier):
                max_inlier = inlier_cnt
                best_H = h_i

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)

        # TODO: 4. apply warping

        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='bl')

    out = dst

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)