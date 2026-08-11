import os

from network.model12 import ConDSeg12
from network.model2 import ConDSeg2
from network.model3 import ConDSeg3
from network.model4 import ConDSeg4
from network.model5 import ConDSeg5
from network.model7 import ConDSeg7
from network.model8 import ConDSeg8



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
from network.model import ConDSeg
from utils.utils import create_dir, seeding
from utils.utils import calculate_metrics
from utils.run_engine import load_data


def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def process_edge(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)

    y_pred = y_pred > 0.001
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def print_score(metrics_score):
    jaccard = metrics_score[0] / len(test_x)  #
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    f2 = metrics_score[5] / len(test_x)
    iou = metrics_score[6] / len(test_x)  # IoU
    dice = metrics_score[7] / len(test_x)  # Dice

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - IoU: {iou:1.4f} - Dice: {dice:1.4f}")


def evaluate(model, save_path, test_x, test_y, size):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():

            mask_pred, fg_pred, bg_pred, uc_pred = model(image)
            p1 = mask_pred

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p1)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            p1 = process_mask(p1)

        cv2.imwrite(f"{save_path}/mask/{name}.png", p1)

    print_score(metrics_score_1)

    with open(f"{save_path}/result.txt", "w") as file:
        file.write(f"Jaccard: {metrics_score_1[0] / len(test_x):1.4f}\n")
        file.write(f"F1: {metrics_score_1[1] / len(test_x):1.4f}\n")
        file.write(f"Recall: {metrics_score_1[2] / len(test_x):1.4f}\n")
        file.write(f"Precision: {metrics_score_1[3] / len(test_x):1.4f}\n")
        file.write(f"Acc: {metrics_score_1[4] / len(test_x):1.4f}\n")
        file.write(f"F2: {metrics_score_1[5] / len(test_x):1.4f}\n")
        file.write(f"IoU: {metrics_score_1[6] / len(test_x):1.4f}\n")
        file.write(f"Dice: {metrics_score_1[7] / len(test_x):1.4f}\n")


if __name__ == "__main__":
    """ Seeding """

    dataset_name = 'ISIC'

    seeding(42)
    size = (224, 224)
    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConDSeg(224, 224)
    # model = ConDSeg_pvt(128, 128)


    model = model.to(device)


    checkpoint_path = "/home/liuyameng/ConDSeg-main/run_files/ISIC/ISIC_None_lr0.0001_20260115-180129/checkpoint.pth"
    #最好的结果
    # checkpoint_path= "/home/liuyameng/ConDSeg-main/run_files/Glas/Glas_None_lr0.0001_20251219-175509/checkpoint.pth"


    # checkpoint_path = "/home/liuyameng/ConDSeg-main/run_files/Glas/Glas_None_lr0.0001_20251219-195541/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "/home/liuyameng/data/{}/".format(dataset_name)
    (train_x, train_y), (test_x, test_y) = load_data(path)

    save_path = f"results/{dataset_name}/MyModel"

    create_dir(f"{save_path}/mask")
    evaluate(model, save_path, test_x, test_y, size)
