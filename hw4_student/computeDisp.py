import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    pad_size = 1
    wndw_size = -1

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    Il_pad = cv2.copyMakeBorder(Il, top=pad_size, bottom=pad_size, left=pad_size,
                                right=pad_size, borderType=cv2.BORDER_CONSTANT, value=0)
    Ir_pad = cv2.copyMakeBorder(Ir, top=pad_size, bottom=pad_size, left=pad_size,
                                right=pad_size, borderType=cv2.BORDER_CONSTANT, value=0)

    Il_bin = np.zeros((9, *Il_pad.shape), dtype=np.uint8)
    Ir_bin = np.zeros((9, *Ir_pad.shape), dtype=np.uint8)
    bin_idx = 0

    # census distance mask
    for x in {-1, 0, 1}:
        for y in {-1, 0, 1}:
            Il_mask = (Il_pad > np.roll(Il_pad, [y, x], axis=(0, 1)))
            Ir_mask = (Ir_pad > np.roll(Ir_pad, [y, x], axis=(0, 1)))

            Il_bin[bin_idx][Il_mask] = 1
            Ir_bin[bin_idx][Ir_mask] = 1

            bin_idx += 1

    # crop padded binary image
    Il_bin = Il_bin[:, pad_size:-pad_size, pad_size:-pad_size]
    Ir_bin = Ir_bin[:, pad_size:-pad_size, pad_size:-pad_size]


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    # cost list
    Il2Ir_cost = np.zeros((max_disp+1, h, w))
    Ir2Il_cost = np.zeros((max_disp+1, h, w))

    for d in range(max_disp+1):
        Il_shift = Il_bin[:, :, d:]
        Ir_shift = Ir_bin[:, :, :w-d]

        cost = np.sum(np.sum(Il_shift ^ Ir_shift, axis=0),
                      axis=2).astype(np.float32)

        Il_cost = cv2.copyMakeBorder(cost, 0, 0, d, 0, cv2.BORDER_REPLICATE)
        Ir_cost = cv2.copyMakeBorder(cost, 0, 0, 0, d, cv2.BORDER_REPLICATE)

        # filtering
        Il2Ir_cost[d, :, :] = xip.jointBilateralFilter(
            Il, Il_cost, wndw_size, 4, 10)
        Ir2Il_cost[d, :, :] = xip.jointBilateralFilter(
            Ir, Ir_cost, wndw_size, 4, 10)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all

    winner_l = np.argmin(Il2Ir_cost, axis=0)
    winner_r = np.argmin(Ir2Il_cost, axis=0)


    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    # consistency check
    lr_check = np.zeros((h, w), dtype=np.float32)
    x, y = np.meshgrid(range(w), range(h))
    d_x = (x - winner_l)  # x distance: x-D_L(x,y)

    # keep positive coordinate
    pos_mask = (d_x >= 0)
    D_L = winner_l[pos_mask]
    D_R = winner_r[y[pos_mask], d_x[pos_mask]]

    # check valid disparity
    val_mask = (D_L == D_R)  # D_L(x,y) = D_R(x-D_L(x,y))
    lr_check[y[pos_mask][val_mask], x[pos_mask][val_mask]] = winner_l[pos_mask][val_mask]

    # Hole filling
    lr_check_pad = cv2.copyMakeBorder(
        lr_check, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=max_disp)

    F_L = np.zeros((h, w), dtype=np.float32)
    F_R = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            Il_idx, Ir_idx = 0, 0

            while lr_check_pad[y, x+1-Il_idx] == 0:
                Il_idx += 1
            while lr_check_pad[y, x+1+Ir_idx] == 0:
                Ir_idx += 1

            F_L[y, x] = lr_check_pad[y, x+1-Il_idx]
            F_R[y, x] = lr_check_pad[y, x+1+Ir_idx]

    # pixel-wise minimum
    filled_labels = np.min((F_L, F_R), axis=0)

    # Weighted median filtering
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), filled_labels, r=10)

    return labels.astype(np.uint8)
