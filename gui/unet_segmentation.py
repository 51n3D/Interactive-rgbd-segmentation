import cv2
import numpy as np
import torch
from segmentation_technique import SegmentationTechnique
from config import IMAGE_SHAPE, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, UNET_BASE
from unet import UNet


class UNetSegmentation(SegmentationTechnique):
    def __init__(self, image_file, disparity_file, model_pth):
        self.image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if self.image.shape != IMAGE_SHAPE:
            self.image = cv2.resize(self.image, IMAGE_SHAPE, interpolation=cv2.INTER_NEAREST)

        self.disparity = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        self.disparity /= self.disparity.max()
        if self.disparity.shape != IMAGE_SHAPE:
            self.disparity = cv2.resize(self.disparity, IMAGE_SHAPE, interpolation=cv2.INTER_NEAREST)

        self.image /= 255

        self.model = UNet(base=UNET_BASE)
        self.model.load_state_dict(torch.load(model_pth, map_location="cpu"))
        self.model.eval()

    def interactions_signal(self, interactions):
        if len(interactions[0]) > 0:
            signal = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            signal[interactions] = 255
            signal = cv2.GaussianBlur(signal, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
            signal = signal / signal.max()

            return signal
        else:
            return np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.float64)

    def segment(self, pos_interactions: tuple, neg_interactions: tuple) -> np.array:
        fpos_interactions = self.interactions_signal(neg_interactions)
        fneg_interactions = self.interactions_signal(pos_interactions)

        cv2.imshow("debug", np.hstack((fpos_interactions, fneg_interactions)))
        #cv2.waitKey(0)

        data = np.dstack((self.image, self.disparity, fneg_interactions, fpos_interactions))
        data = np.array([data])
        data = data.reshape((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
        model_input = torch.tensor(data, dtype=torch.float32)

        with torch.no_grad():
            prediction = self.model.forward(model_input)

        np_prediction = prediction.detach().numpy()
        cv2.imshow("debug_2", np_prediction[0][0])

        np_prediction = np.where(np_prediction > 0.5, 1, 0)

        return np_prediction[0][0]
