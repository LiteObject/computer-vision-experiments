def precision(true_positives, false_positives):
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)

def recall(true_positives, false_negatives):
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)

def mean_average_precision(precisions, recalls):
    if not precisions or not recalls:
        return 0.0
    return sum(precisions) / len(precisions)

def calculate_metrics(predictions, ground_truths):
    true_positives = sum(p == g == 1 for p, g in zip(predictions, ground_truths))
    false_positives = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truths))
    false_negatives = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truths))

    precision_value = precision(true_positives, false_positives)
    recall_value = recall(true_positives, false_negatives)
    mAP = mean_average_precision([precision_value], [recall_value])

    return {
        'precision': precision_value,
        'recall': recall_value,
        'mean_average_precision': mAP
    }