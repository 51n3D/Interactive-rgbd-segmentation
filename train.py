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

MAX_INTERACTIONS = 10
BASE = 3
BATCH_SIZE = 4
DOWNSAMPLE = 4

SINGLE_INSTANCE_LABELS = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33])

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN = 0
VALIDATE = 1


def filter_labels(labels, classes):
    # single occurred labeled instance in the image
    single_filtered = labels[labels < 1000][np.in1d(labels[labels < 1000], classes)]
    # multiple occurred labeled instances in the image
    multiple_normalized = np.floor(labels[labels > 1000] / 1000)
    multiple_filtered = labels[labels > 1000][np.in1d(multiple_normalized, classes)]
    # append together and return
    return np.append([single_filtered], [multiple_filtered])


def total_instances(instances_files: list) -> int:
    count = 0
    for instances_file in instances_files:
        instances = cv2.imread(instances_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        count += filter_labels(np.unique(instances), SINGLE_INSTANCE_LABELS).shape[0]
    return count


def downsample(image: np.ndarray, ratio: int) -> np.ndarray:
    dim = (int(image.shape[1] / ratio), int(image.shape[0] / ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)


def distance_map(shape: tuple, marked_pixels: tuple) -> np.ndarray:
    map = np.zeros(shape, dtype=np.uint8)
    map[marked_pixels] = 1
    return cv2.distanceTransform(map, cv2.DIST_L2, cv2.DIST_MASK_3)


# def guidance_signal_tr(label: int, target: np.ndarray, prediction: np.ndarray) -> tuple:
#     # get indices of false negative/positive pixels
#     error = target - prediction
#     false_neg = np.where(error > 0)
#     false_pos = np.where(error < 0)
#     # calculate distance transform of false negative/positive pixels
#     chamfer_fneg = distance_map(target.shape, false_neg) \
#         if false_neg[0].shape > false_pos[0].shape else np.zeros(target.shape)
#     chamfer_fpos = distance_map(target.shape, false_pos) \
#         if false_pos[0].shape > false_neg[0].shape else np.zeros(target.shape)
#     # prevent owerflow
#     if chamfer_fneg.max() > 700:
#         chamfer_fneg = chamfer_fneg / chamfer_fneg.max() * 700
#     if chamfer_fpos.max() > 700:
#         chamfer_fpos = chamfer_fpos / chamfer_fpos.max() * 700
#     # return converted distance maps to probabilty maps
#     return np.expm1(chamfer_fneg.astype(np.float64)), np.expm1(chamfer_fpos.astype(np.float64))


def dist_to_guidance(chamfer_dist: np.ndarray) -> np.ndarray:
    guidance = np.zeros(chamfer_dist.shape)
    if chamfer_dist.max() == 0:
        return guidance
    guidance[np.unravel_index(np.argmax(chamfer_dist), chamfer_dist.shape)] = 1
    guidance = cv2.GaussianBlur(guidance, (9, 9), 2)
    return guidance / guidance.max()


def guidance_signal_tr(target: np.ndarray, prediction: np.ndarray) -> tuple:
    # get indices of false negative/positive pixels
    error = target - prediction
    false_neg = np.where(error > 0)
    false_pos = np.where(error < 0)
    # calculate distance transform of false negative/positive pixels
    chamfer_fneg = distance_map(target.shape, false_neg) \
        if false_neg[0].shape > false_pos[0].shape else np.zeros(target.shape)
    chamfer_fpos = distance_map(target.shape, false_pos) \
        if false_pos[0].shape > false_neg[0].shape else np.zeros(target.shape)
    # prevent owerflow
    if chamfer_fneg.max() > 700:
        chamfer_fneg = chamfer_fneg / chamfer_fneg.max() * 5000
    if chamfer_fpos.max() > 700:
        chamfer_fpos = chamfer_fpos / chamfer_fpos.max() * 5000
    # convert distance maps to guidance signal
    positive_click = dist_to_guidance(chamfer_fneg) # positive click
    negative_click = dist_to_guidance(chamfer_fpos) # negative click
    return positive_click, negative_click


def pixel_accuracy(target: np.ndarray, prediction: np.ndarray) -> np.float32:
    error = np.abs(target - prediction).sum()
    full = np.prod(target.shape).astype(np.float32)
    return (full - error.sum()) / full
     

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


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def run(model, optimizer, max_interactions, dataset, batch_size, process_type) -> None:
    out = []
    
    image_ending = "leftImg8bit"
    disparity_ending = "disparity"
    instances_ending = "instanceIds"
    
    for city in os.listdir(dataset):
        city_dir = os.path.join(dataset, city)
        log(2, "Current city - " + city)
        # load files
        instances_files = np.sort(np.array(glob.glob(os.path.join(city_dir, "*" + instances_ending + ".png"))))
        disparity_files = np.sort(np.array(glob.glob(os.path.join(city_dir, "*" + disparity_ending + ".png"))))
        image_files = np.sort(np.array(glob.glob(os.path.join(city_dir, "*" + image_ending + ".png"))))
        # select random batch
        n = batch_size if batch_size < instances_files.shape[0] else instances_files.shape[0] - 1
        image_batch = np.random.choice(instances_files.shape[0], n, replace=False)
        # setup progress bar
        progress_bar = tqdm.tqdm(total=n*max_interactions, ncols=100)
        for image_file, instances_file, disparity_file in zip(image_files[image_batch], instances_files[image_batch], disparity_files[image_batch]):
            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(np.float32) # rgb
            # convert disparity to depth map
            disparity = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED).astype(np.float32) # disparity map
            disparity /= disparity.max()
            #disparity[disparity > 0] = (disparity[disparity > 0] - 1.) / 256.
            #depth = (0.209313 * 2262.52) / disparity
            # get instance segmentation of image (to generate new dataset)
            instances = cv2.imread(instances_file, cv2.IMREAD_UNCHANGED).astype(np.uint32)
            # downsample data
            image = downsample(image, DOWNSAMPLE)
            image /= 255.0
            disparity = downsample(disparity, DOWNSAMPLE)
            # label of each instance in the target image (filter single instance labels only)
            instance_lbs = filter_labels(np.unique(instances), SINGLE_INSTANCE_LABELS)
            if instance_lbs.shape[0] == 0:
                progress_bar.update(MAX_INTERACTIONS)
                continue
            # select random batch
            n = batch_size if batch_size < instance_lbs.shape[0] else instance_lbs.shape[0]
            batch = np.random.choice(instance_lbs.shape[0], n, replace=False)
            targets = []
            fpos_guidances = []
            fneg_guidances = []
            for i, i_lb in enumerate(instance_lbs[batch]):
                # single target image for current lable
                #label = i_lb if i_lb < 1000 else math.floor(i_lb / 1000)
                target = downsample(np.where(instances == i_lb, 1, 0), DOWNSAMPLE)
                targets.append(target)
                # generate positive/negative guidance
                fneg_guidance, fpos_guidance = guidance_signal_tr(targets[i], np.zeros(targets[i].shape))
                fneg_guidances.append(fneg_guidance.astype(np.float32))
                fpos_guidances.append(fpos_guidance.astype(np.float32))
            # initialize batch shaped old interaction maps
            np_targets = np.array(targets)
            old_fneg_guidances = np.full(np_targets.shape, 0, dtype=np.float32)
            old_fpos_guidances = np.full(np_targets.shape, 0, dtype=np.float32)
            for current_inter in range(max_interactions):
                data = []
                for b in range(batch.shape[0]):
                    # generate intensity map for positive/negative click
                    w = (max_interactions - current_inter) / float(max_interactions) # weight of click linearly decrease with number of interactions
                    fneg_guidances[b] *= w # weight interaction
                    fpos_guidances[b] *= w # weight interaction
                    # update new interaction map with old interactions
                    fneg_guidances[b] += old_fneg_guidances[b]
                    fpos_guidances[b] += old_fpos_guidances[b]
                    #normalize to (0, 1>
                    if fneg_guidances[b].max() > 0:
                        fneg_guidances[b] /= fneg_guidances[b].max()
                    if fpos_guidances[b].max() > 0:
                        fpos_guidances[b] /= fpos_guidances[b].max()
                    #build 6 channel image for training (rgbdpn)
                    data.append(np.dstack((image, disparity, fneg_guidances[b], fpos_guidances[b])))
                    # update old guidances
                    old_fneg_guidances[b] = fneg_guidances[b].copy()
                    old_fpos_guidances[b] = fpos_guidances[b].copy()
                data = np.array(data)
                # cv2.imshow("rgb", (data[0,:,:,0:3] * 255).astype(np.uint8))
                # cv2.imshow("disparity", (data[0,:,:,3] * 255).astype(np.uint8))
                # cv2.imshow("positive", (data[0,:,:,4] * 255).astype(np.uint8))
                # cv2.imshow("negative", (data[0,:,:,5] * 255).astype(np.uint8))
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
                tensor_targets = np_targets.reshape((np_targets.shape[0], 1, np_targets.shape[1], np_targets.shape[2]))
                gpu_targets = torch.tensor(tensor_targets, dtype=torch.float32).to(device)
                if process_type == TRAIN:
                    gpu_prediction = model.forward(model_input)
                elif process_type == VALIDATE:
                    with torch.no_grad():
                        gpu_prediction = model.forward(model_input)
                np_prediction = gpu_prediction.cpu().detach().numpy()
                # cv2.imshow("prediction", (np_prediction[0][0] * 255).astype(np.uint8))
                np_prediction = np.where(np_prediction >= 0.5, 1, 0)
                # cv2.imshow("prediction - threshold", (np_prediction[0][0] * 255).astype(np.uint8))
                # add new corrections (new pos/neg clicks)
                for b in range(batch.shape[0]):
                    fneg_guidance, fpos_guidance = guidance_signal_tr(np_targets[b], np_prediction[b][0])
                    fneg_guidances[b] = fneg_guidance.astype(np.float64)
                    fpos_guidances[b] = fpos_guidance.astype(np.float64)
                progress_bar.update(1)
                # cv2.waitKey()
                # calculate training metric
                if process_type == TRAIN:
                    # Calculate loss and backpropate
                    #print(prediction)
                    loss = model.backpropagation(gpu_prediction, gpu_targets, optimizer).cpu().detach().numpy()
                    out.append(loss)
                    progress_bar.set_postfix({'loss (batch)': loss})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # calculate validation metrics
            if process_type == VALIDATE:
                for b in range(batch.shape[0]):
                    pa = pixel_accuracy(np_targets[b], np_prediction[b][0])
                    iou = intersection_over_union(np_targets[b], np_prediction[b][0])
                    out.append([pa, iou])
                    progress_bar.set_postfix({'min val in prediction': np_prediction[b][0].min()})
            # show_target_and_prediction_image(prediction, target)  
        progress_bar.close()
    return np.array(out)

        
def main() -> None:
    import sys
    if len(sys.argv) > 2:
        best_model = os.path.join(sys.argv[2], "best_model")
        training_data = os.path.join(sys.argv[2], "training_data")
    else:
        best_model = "best_model"
        training_data = "training_data"
    best_model_path = os.path.join(best_model, "InteractiveModel.pth")
    last_model_path = os.path.join(training_data, "LastModel.pth")
    mean_losses_path = os.path.join(training_data, "mean_losses.npy")
    mean_pixel_acc_path = os.path.join(training_data, "mean_pixel_acc.npy")
    mean_iou_path = os.path.join(training_data, "mean_iou.npy")
    
    log(2, "Training data path: {}".format(training_data))
    log(2, "Best model path: {}".format(best_model))
    log(2, "")
    log(2, "Device - {}".format(device), logger.RED)
    log(2, "")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    max_interactions = MAX_INTERACTIONS # number of max interactions
    model = UNet(base=BASE)
    
    if os.path.exists(last_model_path):
        log(2, "Loading model from {}".format(last_model_path))
        model.load_state_dict(torch.load(last_model_path))
    if not os.path.exists(best_model):
        os.makedirs(best_model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_to(optimizer, device)
    
    if not os.path.exists(training_data):
        os.makedirs(training_data)
         
    log(2, "TRAINING - interactive segmentation of rgbd images", logger.GREEN)
    epoch = 1 if not os.path.exists(mean_losses_path) else np.load(mean_losses_path).shape[0] + 1
    best_miou = -1
    mls = []
    mpas = []
    mious = []
    while True:
        log(2, "")
        log(2, "Epoch " + str(epoch), logger.BLUE)

        if len(sys.argv) > 1:
            dataset = os.path.join(sys.argv[1], "dataset")
        else:
            dataset = "dataset"

        # training
        dataset = os.path.join(dataset, "train")
        log(2, "Dataset path: {}".format(dataset))
        model.train()
        loss = run(model, optimizer, max_interactions, dataset, BATCH_SIZE, TRAIN)
        ml = loss.sum() / loss.shape[0]
        log(2, "Loss (SUM): {}".format(loss.sum()))
        log(2, "Loss (AVG): {}".format(loss.sum()/loss.shape[0]))

        #validation
        log(2, "Validation of Epoch " + str(epoch), logger.YELLOW)
        dataset = os.path.join("dataset", "val")
        model.eval()
        accuracy = run(model, optimizer, max_interactions, dataset, BATCH_SIZE, VALIDATE)
        accuracy = accuracy.T
        mpa = accuracy[0].sum() / accuracy[0].shape[0]
        miou = accuracy[1].sum() / accuracy[1].shape[0]
        log(2, "Mean Pixel Accuracy: {}".format(mpa))
        log(2, "Mean Intersection Over Union: {}".format(miou))

        # save training info
        if os.path.exists(mean_losses_path):
            mls = np.load(mean_losses_path)
        if os.path.exists(mean_pixel_acc_path):
            mpas = np.load(mean_pixel_acc_path)
        if os.path.exists(mean_iou_path):
            mious = np.load(mean_iou_path)
        mls = np.append(ml, mls)
        mpas = np.append(mpa, mpas)
        mious = np.append(miou, mious)
        np.save(mean_losses_path, np.array(mls))
        np.save(mean_pixel_acc_path, np.array(mpas))
        np.save(mean_iou_path, np.array(mious))

        if (miou > best_miou):
            best_miou = miou
            torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), last_model_path)
        
        epoch += 1


if __name__ == "__main__":
    main()
