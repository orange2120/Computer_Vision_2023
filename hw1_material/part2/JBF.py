import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

    # def gaussian(self, x, sigma):
    #     return np.exp(-x**2/sigma**2/2)
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        half_wndw_size = int(self.wndw_size / 2)
 
        output = np.zeros(padded_img.shape)

        # primitive method
        # for x in range(0, img.shape[1]):
        #     print(x)
        #     for y in range(0, img.shape[0]):

        #         num = 0
        #         den = 0
        #         # print(f'(x, y)=({x}, {y})')
        #         # kernel
        #         for i in range(-half_wndw_size, half_wndw_size + 1):
        #             for j in range(-half_wndw_size, half_wndw_size + 1):
        #                 g_s = self.gaussian(i, self.sigma_s)*self.gaussian(j, self.sigma_s)
        #                 g_r = self.gaussian(padded_img[y, x, :] - padded_guidance[y+j, x+i, :], self.sigma_r)
        #                 g = g_s*g_r

        #                 num += g*padded_guidance[y, x, :]
        #                 den += g

        #         output[y, x, :] = num/den

        # vectorized version

        # construct kernels
        # spatial kernel
        g_s = np.exp(-(np.arange(self.pad_w+1)**2)/(2*self.sigma_s**2))
        
        # range kernel (normalized gaussian)
        g_r = np.exp(-((np.linspace(0, 1, 256))**2)/(2*self.sigma_r**2))

        result = np.zeros(padded_img.shape)
        norm_f = np.zeros(padded_img.shape)

        # filtering
        for x in range(-half_wndw_size, half_wndw_size + 1):
            for y in range(-half_wndw_size, half_wndw_size + 1):
        #         print(f'(x, y)=({x}, {y})')
                roll_padd_img = np.roll(padded_img, (y, x), axis=(0, 1))
                roll_padd_guidance = np.roll(padded_guidance, (y, x), axis=(0, 1))

                i_diff = np.abs(roll_padd_guidance-padded_guidance)
                gr_xy = np.product(g_r[i_diff], axis=2) if (padded_guidance.ndim == 3) else g_r[i_diff]
                gs_xy = g_s[np.abs(x)]*g_s[np.abs(y)] # exp(x)*exp(y) = exp(x+y) 

                res = gr_xy*gs_xy

                for ch in range(padded_img.ndim):
                    result[:, :, ch] += np.multiply(roll_padd_img[:,:,ch], res)
                    norm_f[: ,:, ch] += res

        output = result/norm_f
        # crop padded image
        output = output[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, :]

        return np.clip(output, 0, 255).astype(np.uint8)