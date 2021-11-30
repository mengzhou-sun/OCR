import torch as t
import torch.nn.functional as F

#iou_loss(intersection of two Object)
def dice_loss(prediction, target):
    smooth = 1.0
    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):


    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction1 = F.sigmoid(prediction)
    dice = dice_loss(prediction1, target)


    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss




def threshold_predictions_p(predictions, thr=0.5):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds