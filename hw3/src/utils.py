import numpy as np
# import matplotlib.pyplot as plt

def linear_blending(img_left, img_right, img_blended):
    '''
    linear Blending
    '''

    (hr, wr) = img_right.shape[:2]
    img_left_mask = np.zeros((hr, wr), dtype=bool)
    img_right_mask = np.zeros((hr, wr), dtype=bool)
    overlap_mask = np.zeros((hr, wr), dtype=bool)

    # find the left image and right image mask
    left_nonzero = np.nonzero(np.sum(img_left, axis=2))
    right_nonzero = np.nonzero(np.sum(img_right, axis=2))

    img_left_mask[left_nonzero] = 1
    img_right_mask[right_nonzero] = 1

    overlap_mask = np.bitwise_and(img_left_mask, img_right_mask)
    
    # fig = plt.figure()
    # plt.subplot(311)
    # plt.title("Left image")
    # plt.imshow(img_left_mask.astype(int), cmap="gray")
    # plt.subplot(312)
    # plt.title("Right image")
    # plt.imshow(img_right_mask.astype(int), cmap="gray")
    # plt.subplot(313)
    # plt.title("Overlap mask")
    # plt.imshow(overlap_mask.astype(int), cmap="gray")
    # plt.show()
    
    # alpha channel mask
    alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image

    for i in range(hr):
        min_idx = max_idx = -1
        for j in range(wr):
            if (overlap_mask[i, j] and min_idx == -1):
                min_idx = j
            if (overlap_mask[i, j]):
                max_idx = j
        
        if (min_idx == max_idx): # the row's pixels are all zero, or only one pixel not zero
            continue
            
        gradient = 1 / (max_idx - min_idx)
        for j in range(min_idx, max_idx + 1):
            alpha_mask[i, j] = 1 - (gradient * (j - min_idx))

    alpha_mask_left =  np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
    alpha_mask_right =  np.ones((hr, wr, 3)) - alpha_mask_left

    # fig = plt.figure()
    # plt.subplot(211)
    # plt.title("Left alpha channel mask")
    # plt.imshow((alpha_mask_left*255).astype(int), cmap="gray")
    # plt.subplot(212)
    # plt.title("Right alpha channel mask")
    # plt.imshow((alpha_mask_right*255).astype(int), cmap="gray")
    # plt.show()

    # alpha channel blending
    img_blended[overlap_mask] = alpha_mask_left[overlap_mask]*img_left[overlap_mask] + alpha_mask_right[overlap_mask]*img_right[overlap_mask]
    
    return img_blended

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2*N, 9))

    for i in range(N):
        A[2*i  ] = [u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0]*v[i][0], -u[i][1]*v[i][0], -v[i][0]]
        A[2*i+1] = [0, 0, 0, u[i][0], u[i][1], 1, -u[i][0]*v[i][1], -u[i][1]*v[i][1], -v[i][1]]

    _, _, vt = np.linalg.svd(A)

    # TODO: 2.solve H with A
    H = vt[-1].reshape(3, 3) # H is the last column of Vt

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x = np.arange(xmin, xmax)
    y = np.arange(ymin, ymax)
    xv, yv = np.meshgrid(x, y)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xv = xv.flatten()
    yv = yv.flatten()
    one_px = np.ones(xv.shape)

    new_px = np.array([xv, yv, one_px])

    if direction == 'b' or direction == 'bl':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_src_px = np.dot(H_inv, new_px)
        new_src_px = (new_src_px/new_src_px[-1,:])

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (new_src_px[0,:] >= 0) & (new_src_px[0,:] < w_src) & (new_src_px[1,:] >= 0) & (new_src_px[1,:] < h_src)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        val_src_x = new_src_px[0,:][mask].astype(int)
        val_src_y = new_src_px[1,:][mask].astype(int)
        val_dst_x = new_px[0,:][mask].astype(int)
        val_dst_y = new_px[1,:][mask].astype(int)

        # TODO: 6. assign to destination image with proper masking

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_dst_px = np.dot(H, new_px)
        new_dst_px = (new_dst_px/new_dst_px[-1, :]).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (new_dst_px[0,:] >= 0) & (new_dst_px[0,:] < w_dst) & (new_dst_px[1,:] >= 0) & (new_dst_px[1,:] < h_dst)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        val_src_x = new_px[0,:][mask].astype(int)
        val_src_y = new_px[1,:][mask].astype(int)
        val_dst_x = new_dst_px[0,:][mask].astype(int)
        val_dst_y = new_dst_px[1,:][mask].astype(int)

        # TODO: 6. assign to destination image using advanced array indicing

    src_left = np.copy(dst)
    dst[val_dst_y, val_dst_x] = src[val_src_y, val_src_x]

    # apply linear blending
    if direction == 'bl':
        src_right = np.zeros(dst.shape)
        src_right[val_dst_y, val_dst_x] = src[val_src_y, val_src_x]
        out = linear_blending(src_left, src_right, dst)
    else:
        out = dst

    return out