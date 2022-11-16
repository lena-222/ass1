"""
    helper.py includes functions with calculations needed in experiment
"""

def get_mean_per_class(correct_labels, all_labels, num_classes):
    # mean per class accuracy
    not_zero = []
    for j in range(0, num_classes):
        if all_labels[j] > 0:
            not_zero.append(j)
    numerator = [correct_labels[k] / all_labels[k] for k in not_zero]
    return sum(numerator) / len(not_zero)

def calculate_ema(iter_model, iter_ema, ema_rate):
    while True:
        p = next(iter_model, None)
        p_ema = next(iter_ema, None)
        if p is None:
            break
        else:
            p_ema.data = p_ema.data * ema_rate + p.data * (1 - ema_rate)
