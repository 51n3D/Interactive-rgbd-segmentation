import os
import sys
import cv2
import numpy as np
import torch
from segmentation_technique import SegmentationTechnique
from config import IMAGE_SHAPE, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, UNET_BASE

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from unet import UNet, UNetRGBD


class UNetSegmentation(SegmentationTechnique):
    def __init__(self, image_file, disparity_file, model_pth, debug_mode=False):
        self.image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if self.image.shape != IMAGE_SHAPE:
            self.image = cv2.resize(self.image, IMAGE_SHAPE, interpolation=cv2.INTER_NEAREST)

        self.disparity = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        self.disparity /= self.disparity.max()
        if self.disparity.shape != IMAGE_SHAPE:
            self.disparity = cv2.resize(self.disparity, IMAGE_SHAPE, interpolation=cv2.INTER_NEAREST)

        self.image /= 255

        self.model = UNetRGBD(base=UNET_BASE)
        self.model.load_state_dict(torch.load(model_pth, map_location="cpu"))
        self.model.eval()

        self.debug_mode = debug_mode

    def interactions_signal(self, interactions):
        if len(interactions[0]) > 0:
            guidance = np.ones((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            guidance[interactions] = 0
            #signal = cv2.GaussianBlur(signal, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
            #signal = signal / signal.max()

            guidance = cv2.distanceTransform(guidance, cv2.DIST_L2, cv2.DIST_MASK_3)
            guidance /= guidance.max()
            return 1 - guidance
        else:
            return np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.float32)

    def segment(self, pos_interactions: tuple, neg_interactions: tuple) -> np.array:
        fpos_interactions = self.interactions_signal(neg_interactions)
        fneg_interactions = self.interactions_signal(pos_interactions)

        if self.debug_mode:
            cv2.imshow("debug - neg/pos interactions", np.hstack((fpos_interactions, fneg_interactions)))

        #data = np.dstack((self.image, self.disparity, fneg_interactions, fpos_interactions))
        #data = np.array([data])
        #data = data.reshape((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
        #model_input = torch.tensor(data, dtype=torch.float32)

        #rgb_data = self.image.copy()
        #depth_data = np.dstack((self.disparity.copy(), fneg_interactions, fpos_interactions))
        rgb_data = np.dstack((self.image.copy(), fneg_interactions, fpos_interactions))
        depth_data = self.disparity.copy()
        rgb_data = np.array([rgb_data])
        depth_data = np.array([depth_data])

        rgb_data = rgb_data.reshape((rgb_data.shape[0], rgb_data.shape[3], rgb_data.shape[1], rgb_data.shape[2]))
        #depth_data = depth_data.reshape((depth_data.shape[0], depth_data.shape[3], depth_data.shape[1], depth_data.shape[2]))
        depth_data = depth_data.reshape(
            (depth_data.shape[0], 1, depth_data.shape[1], depth_data.shape[2]))
        model_input = (
            torch.tensor(rgb_data, dtype=torch.float32),
            torch.tensor(depth_data, dtype=torch.float32)
        )

        with torch.no_grad():
            prediction = self.model.forward(model_input)

        np_prediction = prediction.detach().numpy()

        if self.debug_mode:
            cv2.imshow("debug_2 - prediction", np_prediction[0][0])

        np_prediction = np.where(np_prediction > 0.5, 1, 0)

        return np_prediction[0][0]
