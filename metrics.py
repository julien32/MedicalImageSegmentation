from sklearn import metrics
import numpy as np




def compute_binary_segmentation_metrics(predictions, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        predictions: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """

    if isinstance(predictions, list):
        predictions = np.stack(predictions)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_predictions = predictions.ravel().astype(np.uint8)
    flat_ground_truth_masks = ground_truth_masks.ravel().astype(np.uint8)


    jaccard_score = metrics.jaccard_score(flat_ground_truth_masks, flat_predictions)
    segmentation_f1_score = metrics.f1_score(flat_ground_truth_masks, flat_predictions)
        
        
    return {
        "jaccard_score": jaccard_score,
        "segmentation_f1_score": segmentation_f1_score,
    }


def compute_binary_classification_metrics(
    predictions, classification_ground_truth
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        predictions: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        classification_ground_truth: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    accuracy = metrics.accuracy_score(classification_ground_truth, predictions)
    f1 = metrics.f1_score(classification_ground_truth, predictions)
    confusion_matrix = metrics.confusion_matrix(classification_ground_truth, predictions, labels=[0,1], normalize=None)

    return {
            "class_acc": accuracy,
            "class_f1": f1,
            "confusion_matrix": confusion_matrix,
            }
