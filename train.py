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
import logger
from logger import log

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN = 0
VALIDATE = 1

def total_instances(instances_files: list) -> int:
    count = 0
    for instances_file in instances_files:
        instances = cv2.imread(instances_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        labels = np.unique(instances)
        count += labels[(labels != 0) | (labels != -1)].shape[0]
    return count


def down_sample(image: np.ndarray, ratio: int) -> np.ndarray:
    dim = (int(image.shape[1] / ratio), int(image.shape[0] / ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)


def distance_map(shape: tuple, marked_pixels: tuple) -> np.ndarray:
    map = np.zeros(shape, dtype=np.uint8)
    map[marked_pixels] = 1
    return cv2.distanceTransform(map, cv2.DIST_L2, cv2.DIST_MASK_3)


def guidance_signal_tr(label: int, target: np.ndarray, prediction: np.ndarray) -> tuple:
    # get indices of false negative/positive pixels
    error = target - prediction
    false_neg = np.where(error > 0)
    false_pos = np.where(error < 0)
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


def pixel_accuracy(target: np.ndarray, prediction: np.ndarray) -> np.float32:
    error = np.abs(target - prediction).sum()
    return error.sum() / np.prod(target.shape).astype(np.float32)
     

def intersection_over_union(target: np.ndarray, prediction: np.ndarray) -> np.float32:
    intersection = target * prediction
    union = (target + prediction).astype(np.float32)
    return intersection.sum() / union.sum()


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


def run(model, optimizer, max_interactions, dataset, batch_size, process_type) -> None:
    out = []
    
    image_ending = "leftImg8bit"
    disparity_ending = "disparity"
    instances_ending = "instanceIds"
    
    for city in os.listdir(dataset):
        city_dir = os.path.join(dataset, city)
        log(2, "Current city - " + city)
        # load files
        instances_files = np.array(glob.glob(os.path.join(city_dir, "*" + instances_ending + ".png")))
        disparity_files = np.array(glob.glob(os.path.join(city_dir, "*" + disparity_ending + ".png")))
        image_files = np.array(glob.glob(os.path.join(city_dir, "*" + image_ending + ".png")))
        # select random batch
        n = batch_size if batch_size < instances_files.shape[0] else instances_files.shape[0] - 1
        batch = np.random.choice(instances_files.shape[0], n, replace=False)
        # setup progress bar
        progress_bar = tqdm.tqdm(total=n*max_interactions, ncols=100)
        for image_file, instances_file, disparity_file in zip(image_files[batch], instances_files[batch], disparity_files[batch]):
            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(np.float32) # rgb
            # convert disparity to depth map
            disparity = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED).astype(np.float32) # disparity map
            #disparity[disparity > 0] = (disparity[disparity > 0] - 1.) / 256.
            #depth = (0.209313 * 2262.52) / disparity
            # get instance segmentation of image (to generate new dataset)
            instances = cv2.imread(instances_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # downsample data
            image = down_sample(image, 2)
            disparity = down_sample(disparity, 2)
            instances = down_sample(instances, 2)
            # label of each instance in the target image (0 - unlabled, -1 - license plate)
            instance_lbs = np.unique(instances)
            instance_lbs = instance_lbs[(instance_lbs != 0) | (instance_lbs != -1)]
            # select random batch
            n = batch_size if batch_size < instance_lbs.shape[0] else instance_lbs.shape[0] - 1
            batch = np.random.choice(instance_lbs.shape[0], n, replace=False)
            targets = []
            fpos_guidances = []
            fneg_guidances = []
            for i, i_lb in enumerate(instance_lbs[batch]):
                # single target image for current lable
                #label = i_lb if i_lb < 1000 else math.floor(i_lb / 1000)
                targets.append(np.where(instances == i_lb, 1, 0))
                # generate positive/negative guidance
                fneg_guidance, fpos_guidance = guidance_signal_tr(i_lb, targets[i], np.zeros(targets[i].shape))
                fneg_guidances.append(fneg_guidance)
                fpos_guidances.append(fpos_guidance)
            # initialize batch shaped old interaction maps
            np_targets = np.array(targets)
            old_fneg_inters = np.full(np_targets.shape, 0)
            old_fpos_inters = np.full(np_targets.shape, 0)
            for current_inter in range(max_interactions):
                data = []
                for b in range(batch.shape[0]):
                    # generate intensity map for positive/negative click
                    #print(fneg_guidances[b], fpos_guidances[b])
                    w = (max_interactions - current_inter) / max_interactions # weight of click linearly decrease with number of interactions
                    fneg_interactions = (fneg_guidances[b] / fneg_guidances[b].max()) * w          \
                        if fneg_guidances[b].max() > 0                                             \
                        else np.zeros(np_targets[b].shape) # normalize to <0, 1>
                    fpos_interactions = (fpos_guidances[b] / fpos_guidances[b].max()) * w          \
                        if fpos_guidances[b].max() > 0                                             \
                        else np.zeros(np_targets[b].shape) # normalize to <0, 1>
                    # update new interaction map with old interactions
                    fneg_interactions += old_fneg_inters[b]
                    fpos_interactions += old_fpos_inters[b]
                    if fneg_interactions.max() > 0:
                        fneg_interactions /= fneg_interactions.max() # normalize back to <0, 1>
                    if fpos_interactions.max() > 0:
                        fpos_interactions /= fpos_interactions.max() # normalize back to <0, 1>
                    # build 6 channel image for training (rgbdpn)
                    data.append(np.dstack((image, disparity, fneg_interactions, fpos_interactions)))
                data = np.array(data)
                #####################################################################################
                # TRAIN model # FILL                                                                #
                # call the model.fit() or model.predict() or whatever                               #
                # - there is no loss computation or parameter update of the model in this for cycle #
                # - loss is being computed from final prediction when the last interaction occured  #
                # (after this cycle compute the loss)                                               #
                #####################################################################################
                # push data to device
                data = data.reshape((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
                model_input = torch.tensor(data, dtype=torch.float32).to(device)
                targets = np_targets.reshape((np_targets.shape[0], 1, np_targets.shape[1], np_targets.shape[2]))
                targets = torch.tensor(targets, dtype=torch.float32)
                if process_type == TRAIN:
                    prediction = model.forward(model_input)
                elif process_type == VALIDATE:
                    with torch.no_grad():
                        prediction = model.forward(model_input)
                prediction = prediction.cpu()
                np_prediction = prediction.detach().numpy()
                np_prediction = np.where(np_prediction > 0.5, 1, 0)
                        
                # add new corrections (new pos/neg clicks)
                for b in range(batch.shape[0]):
                    fneg_guidance, fpos_guidance = guidance_signal_tr(i_lb, np_targets[b], np_prediction[b][0])
                    fneg_guidances[b] = fneg_guidance
                    fpos_guidances[b] = fpos_guidance
                progress_bar.update(1)
            # calculate validation metrics
            if process_type == VALIDATE:
                for b in range(batch.shape[0]):
                    pa = pixel_accuracy(np_targets[b], np_prediction[b][0])
                    iou = intersection_over_union(np_targets[b], np_prediction[b][0])
                    out.append([pa, iou])
            # calculate training metric
            if process_type == TRAIN:
                # Calculate loss and backpropate
                out.append(model.backpropagation(prediction, targets, optimizer).detach().numpy())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # show_target_and_prediction_image(prediction, target)  
        progress_bar.close()
    return np.array(out)

        
def main() -> None:
    log(2, "Device - {}".format(device), logger.RED)
    log(2, "")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    max_interactions = 10 # number of max interactions
    model = UNet(base=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    best_model_path = os.path.join("best_model", "InteractiveModel.pth")
    mean_losses_path = os.path.join("training_data", "mean_losses")
    mean_pixel_acc_path = os.path.join("training_data", "mean_pixel_acc")
    mean_iou_path = os.path.join("training_data", "mean_iou")

    log(2, "TRAINING - interactive segmentation of rgbd images", logger.GREEN)
    epoch = 1
    best_miou = 0
    mls = []
    mpas = []
    mious = []
    while True:
        log(2, "")
        log(2, "Epoch " + str(epoch), logger.BLUE)

        # training
        dataset = os.path.join("dataset", "train")
        model.train()
        loss = run(model, optimizer, max_interactions, dataset, 32, TRAIN)
        ml = loss.sum() / loss.shape[0]
        log(2, "Loss (SUM): {}".format(loss.sum()))
        log(2, "Loss (AVG): {}".format(loss.sum()/loss.shape[0]))

        #validation
        log(2, "Validation of Epoch " + str(epoch), logger.YELLOW)
        dataset = os.path.join("dataset", "val")
        model.eval()
        accuracy = run(model, optimizer, max_interactions, dataset, 32, VALIDATE)
        accuracy = accuracy.T
        mpa = accuracy[0].sum() / accuracy[0].shape[0]
        miou = accuracy[1].sum() / accuracy[1].shape[0]
        log(2, "Mean Pixel Accuracy: {}".format(mpa))
        log(2, "Mean Intersection Over Union: {}".format(miou))

        # save training info
        mls.append(ml)
        mpas.append(mpa)
        mious.append(miou)
        np.save(mean_losses_path, np.array(mls))
        np.save(mean_pixel_acc_path, np.array(mpas))
        np.save(mean_iou_path, np.array(mious))

        if (miou > best_miou):
            best_miou = miou
            torch.save(model.state_dict(), best_model_path)

        epoch += 1


if __name__ == "__main__":
    main()
