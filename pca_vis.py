import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms as T
from math import sqrt

def pca_vis(feature_maps_fit_data):        # (c, h, w)
    feature_maps_fit_data = feature_maps_fit_data.reshape(feature_maps_fit_data.shape[0], -1).t()
    feature_maps_fit_data = feature_maps_fit_data.detach().cpu().numpy()
    feature_maps_transform_data = feature_maps_fit_data
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data)  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(1, -1, 3)  # B x (H * W) x 3

    pca_img = feature_maps_pca[0]  # (H * W) x 3
    h = w = int(sqrt(pca_img.shape[0]))
    pca_img = pca_img.reshape(h, w, 3)
    pca_img_min = pca_img.min(axis=(0, 1))
    pca_img_max = pca_img.max(axis=(0, 1))
    pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
    return pca_img
    # pca_img = PIL.Image


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     # model arguments
#     parser.add_argument('--feature_map', type=str, default="",
#                         help='feature path')
#     parser.add_argument('--save_dir', type=str, default="",
#                         help='feature path save dir')
#     args = parser.parse_args()
#     main(args) 

