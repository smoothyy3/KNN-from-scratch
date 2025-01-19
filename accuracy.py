def accuracy(y_true, y_pred):
    if len(y_pred) != len(y_true):
        raise ValueError("Prediciton and ground truth must have the same length!")
    
    # count false if needed for other methods (like f1 score, precision...)
    true = 0
    false = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            true += 1
        else:
            false += 1
    
    acc = true / len(y_pred)
    return acc
    