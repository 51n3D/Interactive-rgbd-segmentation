import os, glob
import numpy as np
import imageio
import math
import cv2
import tqdm
from unet import UNet
import torch
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WHITE = 37
GREEN = 92
BLUE = 96
YELLOW = 93


def log(level: int, message: str, color: int = WHITE):
    log_levels = {0: "TRACE", 1: "DEBUG", 2: "INFO",
        3: "WARN", 4: "ERROR", 5: "FATAL"}
    print("[{}]:\33[{}m".format(log_levels[level], color), message, "\33[{}m".format(WHITE))


def total_instances(instances_files: list) -> int:
    count = 0
    for instances_file in instances_files:
        instances = imageio.imread(instances_file)
        count += np.unique(instances).shape[0]
    return count


def distance_map(shape: tuple, marked_pixels: tuple) -> np.ndarray:
    map = np.zeros(shape, dtype=np.uint8)
    map[marked_pixels] = 1
    return cv2.distanceTransform(map, cv2.DIST_L2, cv2.DIST_MASK_3)


def guidance_signal_tr(label: int, target: np.ndarray, prediction: np.ndarray) -> tuple:
    # get indices of false negative/positive pixels
    error = target - prediction
    false_neg = np.where((error != -label) & (error != 0))
    false_pos = np.where(error == -label)
    # calculate distance transform of false negative/positive pixels
    chamfer_fneg = distance_map(target.shape, false_neg)
    chamfer_fpos = distance_map(target.shape, false_pos)
    # prevent owerflow
    if chamfer_fneg.max() > 700:
        chamfer_fneg = chamfer_fneg / chamfer_fneg.max() * 700
    if chamfer_fpos.max() > 700:
        chamfer_fpos = chamfer_fpos / chamfer_fpos.max() * 700
    # return converted distance maps to probabilty maps
    return np.expm1(chamfer_fneg.astype(np.float64)), np.expm1(chamfer_fpos.astype(np.float64))

def show_target_and_prediction_image(prediction, target):
    f, axarr = plt.subplots(1, 2)
    axarr[0] = plt.imshow(prediction.detach().numpy()[0][0])
    axarr[1] = plt.imshow(target)
    plt.show()

def prepare_image_data_for_model(image, fneg_interactions, fpos_interactions, disparity):
    image = image.reshape((1, image.shape[2], image.shape[0], image.shape[1])).astype("double")
    fneg_interactions = fneg_interactions.reshape((1, 1, fneg_interactions.shape[0], fneg_interactions.shape[1])).astype("double")
    fpos_interactions = fpos_interactions.reshape((1, 1, fpos_interactions.shape[0], fpos_interactions.shape[1])).astype("double")
    disparity = disparity.reshape((1, 1, disparity.shape[0], disparity.shape[1])).astype("double")
    return image, fneg_interactions, fpos_interactions, disparity

def main() -> None:
    max_interactions = 5 # number of max interactions
    model = UNet(base=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    image_ending = "leftImg8bit"
    disparity_ending = "disparity"
    instances_ending = "instanceIds"

    log(2, "TRAINING - interactive segmentation of rgbd images", GREEN)
    epoch = 1
    while True:
        log(2, "")
        log(2, "Epoch " + str(epoch), BLUE)
        train_dir = os.path.join("dataset", "train")
        for city in os.listdir(train_dir):
            city_dir = os.path.join(train_dir, city)
            log(2, "Current city - " + city)
            instances_files = glob.glob(os.path.join(city_dir, "*" + instances_ending + ".png"))
            disparity_files = glob.glob(os.path.join(city_dir, "*" + disparity_ending + ".png"))
            image_files = glob.glob(os.path.join(city_dir, "*" + image_ending + ".png"))
            progress_bar = tqdm.tqdm(total=total_instances(instances_files), ncols=100)
            for image_file, instances_file, disparity_file in zip(image_files, instances_files, disparity_files):
                image = imageio.imread(image_file) # rgb
                disparity = imageio.imread(disparity_file) # depth map
                # get instance segmentation of image (to generate new dataset)
                instances = imageio.imread(instances_file)
                # label of each instance in the target image
                instance_lbs = np.unique(instances)
                for i_lb in instance_lbs:
                    # 0 - unlabled, -1 - license plate
                    if i_lb == 0 or i_lb == -1:
                        progress_bar.update(1)
                        continue
                    # extract single target image for current lable
                    label = i_lb if i_lb < 1000 else math.floor(i_lb / 1000)
                    target = np.where(instances == i_lb, label, 0)
                    # generate positive/negative guidance
                    fneg_guidance, fpos_guidance = guidance_signal_tr(i_lb, target, np.zeros(target.shape))
                    # initialize old interaction maps
                    old_fneg_inter = np.zeros(target.shape)
                    old_fpos_inter = np.zeros(target.shape)
                    # initialize prediction (loss is computed after for cycle)
                    for i in range(max_interactions):
                        # generate intensity map for positive/negative click 
                        w = (max_interactions - i) / max_interactions # weight of click linearly decrease with number of interactions
                        fneg_interactions = (fneg_guidance / fneg_guidance.max()) * w # this will never be zero, first interaction is always needed
                        fpos_interactions = (fpos_guidance / fpos_guidance.max()) * w if fpos_guidance.max() > 0 else np.zeros(target.shape) # fpos_guidance.max() is zero at first iteration
                        # update new interaction map with old interactions
                        fneg_interactions += old_fneg_inter
                        fpos_interactions += old_fpos_inter
                        fneg_interactions /= fneg_interactions.max() # normalize back to <0, 1> (this will never be zero, first interaction is always needed)
                        if fpos_interactions.max() > 0: # zero at first iteration
                            fpos_interactions /= fpos_interactions.max() # normalize back to <0, 1>
                        # build 6 channel image for training (rgbfnegfpos - model_input[:,:,:-2], disparity - model_input[:,:,-2], target - model_input[:,:,-1])
                        model_input = np.dstack((image, fneg_interactions, fpos_interactions, disparity))
                        # model_input = prepare_image_data_for_model(image, fneg_interactions, fpos_interactions, disparity)
                        #####################################################################################
                        # TRAIN model # FILL                                                                #
                        # call the model.fit() or model.predict() or whatever                               #
                        # - there is no loss computation or parameter update of the model in this for cycle #
                        # - loss is being computed from final prediction when the last interaction occured  #
                        # (after this cycle compute the loss)                                               #
                        #####################################################################################
                        prediction = model.forward(model_input)
                        np_prediction = prediction.detach().numpy()[0][0]
                        # add new corrections (new pos/neg clicks)
                        fneg_guidance, fpos_guidance = guidance_signal_tr(i_lb, target, np_prediction)

                    # Calculate loss and backpropate
                    model.backpropagation(prediction, target, optimizer)
                    # show_target_and_prediction_image(prediction, target)
                    progress_bar.update(1)

            progress_bar.close()
        
        #####################################################################################
        # VALIDATE model # FILL                                                             #
        #####################################################################################
        log(2, "Validation of Epoch " + str(epoch), YELLOW)

        epoch += 1


if __name__ == "__main__":
    main()
