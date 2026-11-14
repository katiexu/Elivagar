import numpy as np

def mse_batch_loss(preds, labels, mean=True):
    """
    MSE sample loss for a batch of predictions and labels. Use only with 1-dimensional predictions.
    """
    losses = np.power(np.subtract(preds, labels), 2)

    if mean:
        losses = np.mean(losses)
        
    return losses 


def mse_vec_batch_loss(preds, labels):
    """
    MSE loss for a batch of predictions and labels. Use only with 2-dimensional batch of predictions.
    """
    return np.mean(np.sum(np.power(np.subtract(preds, labels), 2), 1))


def batch_acc(preds, labels):
    """
    Accuracy for a batch of predictions and labels. Use only with 1-dimensional predictions.
    """    
    return np.mean(mse_batch_loss(preds, labels, False) < 1)


def vec_batch_acc(preds, labels):
    """
    Accuracy for a batch of predictions and labels. Use nly with 2-dimensional batch of predictions.
    """
    return np.mean(np.sum(np.multiply(preds, labels) > 0, 1) == preds.shape[1])


def ce_vec_batch_acc(logits, labels):
    """Classification accuracy for multiclass logits and integer class labels."""
    labels = np.asarray(labels).reshape(-1).astype(int)
    return np.mean(np.argmax(logits, axis=1) == labels)


def nll_vec_batch_loss(logits, labels):
    """Mean negative log-likelihood computed stably from multiclass logits."""
    logits = np.asarray(logits)
    labels = np.asarray(labels).reshape(-1).astype(int)
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted_logits), axis=1))
    return np.mean(log_normalizer - shifted_logits[np.arange(len(labels)), labels])