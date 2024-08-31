import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from logging_utils import logger

def plot_precision_recall_curve(true_labels, scores, task_name, embedding_dim, num_epochs, batch_size):
    precision, recall, thresholds_pr = precision_recall_curve(true_labels, -scores)
    pr_auc = auc(recall, precision)
    plt.figure()

    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    logger.info(f'PR AUC: {pr_auc}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PRC_{task_name}_{embedding_dim}_{num_epochs}_batch{batch_size}')
    plt.legend()
    plt.savefig(f'PRC_{task_name}_{embedding_dim}_{num_epochs}_batch{batch_size}.png')
    plt.show()

def plot_roc_curve(true_labels, scores, task_name, embedding_dim, num_epochs, batch_size):
    fpr, tpr, thresholds_roc = roc_curve(true_labels, -scores)
    roc_auc = auc(fpr, tpr)
    logger.info(f'ROC AUC: {roc_auc}')
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC_{task_name}_{embedding_dim}_{num_epochs}_batch{batch_size}')
    plt.legend()
    plt.savefig(f'ROC_{task_name}_{embedding_dim}_{num_epochs}_batch{batch_size}.png')
    plt.show()
