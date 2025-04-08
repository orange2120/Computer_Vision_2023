import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_gaussian_images(self, image, output):
        output.append(image)
        for sig in range(1, self.num_guassian_images_per_octave):
            gs_img = cv2.GaussianBlur(image, (0, 0), self.sigma**sig)
            output.append(gs_img)

    def get_dog_images(self, images, output):
        for i in range(0, self.num_DoG_images_per_octave):
            dog = cv2.subtract(images[i+1], images[i])
            output.append(dog)

    # index from (0, 0, 0) to (2, 2, 2)
    def is_local_extremum(self, cubic):
        'find local extremum in 3x3x3 cubic'

        pixel = cubic[1][1, 1]
        if abs(pixel) >= self.threshold:
            # print(pixel)
            if pixel > 0:  # check local maxima
                return (pixel >= cubic[0]).all() and (pixel >= cubic[2]).all() and (pixel >= cubic[1][0, :]).all() \
                    and (pixel >= cubic[1][2, :]).all() and pixel >= cubic[1][1, 0] and pixel >= cubic[1][1, 2]
            elif pixel < 0:  # check local minima
                return (pixel <= cubic[0]).all() and (pixel <= cubic[2]).all() and (pixel <= cubic[1][0, :]).all() \
                    and (pixel <= cubic[1][2, :]).all() and pixel <= cubic[1][1, 0] and pixel <= cubic[1][1, 2]
        return False

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)

        gaussian_images = []

        for i in range(0, self.num_octaves):
            imgs = []
            self.get_gaussian_images(image, imgs)
            gaussian_images.append(imgs)

            # down-sample the last image in first octave to 1/2 size
            image = cv2.resize(imgs[-1], (0, 0), fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_NEAREST)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []

        for oct_gaussian_images in gaussian_images:
            dog = []
            self.get_dog_images(oct_gaussian_images, dog)
            dog_images.append(dog)

        # print(f'DoG images= {len(dog_images[0])}')

        # write DoG images
        # for i, dogs in enumerate(dog_images):
        #     for j, img in enumerate(dogs):
        #         norm_img = (img - img.min())/(img.max() - img.min())*255.0
        #         cv2.imwrite(f'./output/DoG{i + 1}-{j + 1}.png', norm_img)


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        keypoints = []

        # extract images per octave
        for oct_idx, oct_dog in enumerate(dog_images):
            # print(f'octaves = {oct_idx}')
            # extract every 3 DoG images
            for i in range(1, self.num_DoG_images_per_octave - 1):
                # print(f'    img in octave = {i}')
                for x in range(1, oct_dog[0].shape[0] - 1):
                    for y in range(1, oct_dog[0].shape[1] - 1):
                        if self.is_local_extremum([oct_dog[i-1][x-1:x+2, y-1:y+2], oct_dog[i][x-1:x+2, y-1:y+2], oct_dog[i+1][x-1:x+2, y-1:y+2]]):
                            keypoints.append([(2**oct_idx)*x, (2**oct_idx)*y])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique

        # print(f'Original keypoints = {len(keypoints)}')
        keypoints = np.array(keypoints)
        keypoints = np.unique(keypoints, axis=0)
        # print(keypoints)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]

        # print(f'Total keypoints = {len(keypoints)}')
        return keypoints
