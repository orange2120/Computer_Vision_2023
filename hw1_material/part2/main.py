import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###

    rgb_weights = []

    # read setting file
    setting_file = open(args.setting_path, 'r')
    setting = setting_file.readlines()
    for i in range(1, 6):
        rgb_weights.append(list(map(float, setting[i].split(','))))

    sigma_s = int(setting[6].split(',')[1])
    sigma_r = float(setting[6].split(',')[-1])

    print(rgb_weights)
    print(f's_s = {sigma_s}, s_r = {sigma_r}')

    # generate guidance images
    guidances = []
    guidances_jbf = []

    for weights in rgb_weights:
        m = np.array(weights).reshape((1,3))
        y = cv2.transform(img_rgb, m)
        guidances.append(y)

    # for y in guidances:
    #     cv2.imshow('image', y)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)

    gray_cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    # cv2.imshow('BF out', bf_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    jbf_out = cv2.cvtColor(jbf_out,cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f'./output/BF_{args.image_path[-5]}.png', bf_out)
    # cv2.imwrite(f'./output/JBF_CVT_{args.image_path[-5]}.png', jbf_out)
    # cv2.imwrite(f'./output/gray_CVT_{args.image_path[-5]}.png', img_gray)
    print(f'BGR2GRAY = {gray_cost}')

    for idx, img in enumerate(guidances):
        gray_jbf = JBF.joint_bilateral_filter(img_rgb, img)
        cost = np.sum(np.abs(bf_out.astype('int32')-gray_jbf.astype('int32')))
        gray_jbf = cv2.cvtColor(gray_jbf,cv2.COLOR_RGB2BGR)
        guidances_jbf.append(gray_jbf)
        path_suffix = args.setting_path.split('/')[-1][:-4]
        # cv2.imwrite(f'./output/gray_{path_suffix}_{idx}.png', img)
        # cv2.imwrite(f'./output/JBF_RGB_{path_suffix}_{idx}.png', gray_jbf)
        print(f'[{idx}] cost = {cost}')



if __name__ == '__main__':
    main()