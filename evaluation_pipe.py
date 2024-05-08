import itertools
import os
import random
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.metrics import roc_curve, auc


def classification_evaluation_pipeline(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list
) -> None:
    """
    Evaluates the classification model by generating a comprehensive report including classification metrics,
    confusion matrix, and ROC curve.

    Args:
        y_true (np.ndarray): True labels of the test data.
        y_pred (np.ndarray): Predicted labels as returned by the classifier.
        y_prob (np.ndarray): Probabilities of the positive class or decision function values required for ROC curve calculation.
        classes (list): List of class names for more interpretable visualizations.

    Example usage:
        y_pred = model.predict(X_test)
        classes = ["Class 0", "Class 1"]
        classification_evaluation_pipeline(y_true=y_test, y_pred=y_pred, classes=classes)
    """
    print("1. Printing Classification Report")
    print(classification_report(y_pred=y_pred, y_true=y_true))
    print("2. Plotting Model Metrics")
    result = calculate_model_metrics(y_pred=y_pred, y_true=y_true)
    print("3. Plot Confusion Matrix")
    make_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=classes)


def calculate_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
    figsize: tuple[int, int] = (10, 10),
    color: str = "blue",
) -> dict:
    """
    This function uses the 'accuracy_score' and 'precision_recall_fscore_support' from scikit-learn to calculate
    the accuracy, precision, recall, and f1-score of a classification model, with metrics returned as percentages.
    It also plots these metrics in a bar chart, using Matplotlib, with customizable figure size and bar color.

    Args:
        y_true (np.ndarray): Array of truth labels (must be same shape as y_pred).
        y_pred (np.ndarray): Array of predicted labels (must be same shape as y_true).
        average (str, optional): The strategy for averaging. Can be one of 'micro', 'macro', 'samples', 'weighted', or 'binary'.
                                 Defaults to 'weighted' which accounts for label imbalance by computing the average of binary metrics
                                 in which each class's score is weighted by its presence in the true data sample.
        figsize (tuple[int, int]): Size of output figure (default=(10, 10)).
        color (str, optional): Color of the bars in the plot. Can be a single color format string, or a sequence of color
                               specifications of length equal to the number of bars. Accepts name of a color (e.g., 'blue', 'green'),
                               hex string (e.g., '#008000'), RGB tuple (e.g., (0,1,0)), or grayscale intensity (e.g., '0.5').
                               Defaults to 'blue'. Example colors: 'red', '#FFDD44', (0.1, 0.2, 0.5), '0.75'.

    Returns:
        dict[str, float]: A dictionary containing the accuracy, precision, recall, and f1-score, each as a percentage.

    Example usage:
        model_metrics = calculate_model_metrics(y_true, y_pred)
        model_metrics_custom_color = calculate_model_metrics(y_true, y_pred, color='green')
        model_metrics_rgb_color = calculate_model_metrics(y_true, y_pred, color=(0.5, 0.2, 0.8))
    """
    # Calculate model accuracy and convert to percentage
    model_accuracy = accuracy_score(y_true, y_pred)

    # Calculate model precision, recall, and f1 score using specified average method and convert to percentages
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )

    model_results = {
        "accuracy": model_accuracy * 100,
        "precision": model_precision * 100,
        "recall": model_recall * 100,
        "f1": model_f1 * 100,
    }

    plt.figure(figsize=figsize)
    plt.bar(model_results.keys(), model_results.values(), color=color)
    plt.title("Model Metrics")
    plt.xlabel("Metric Names")
    plt.ylabel("Metric Values in %")
    plt.show()

    return model_results


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
    savefig: bool = False,
) -> None:
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels, with options to normalize
    and save the figure.

    Args:
      y_true (np.ndarray): Array of truth labels (must be same shape as y_pred).
      y_pred (np.ndarray): Array of predicted labels (must be same shape as y_true).
      classes (np.ndarray): Array of class labels (e.g., string form). If `None`, integer labels are used.
      figsize (tuple[int, int]): Size of output figure (default=(10, 10)).
      text_size (int): Size of output figure text (default=15).
      norm (bool): If True, normalize the values in the confusion matrix (default=False).
      savefig (bool): If True, save the confusion matrix plot to the current working directory (default=False).

    Returns:
        None: This function does not return a value but displays a Confusion Matrix. Optionally, it saves the plot.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10,
                            norm=True,
                            savefig=True)
    """
    # Create the confusion matrix
    cm = (
        confusion_matrix(y_true, y_pred, normalize="true")
        if norm
        else confusion_matrix(y_true, y_pred)
    )

    # Plot the figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    # Set class labels
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(len(cm))

    # Set the labels and titles
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Annotate the cells with the appropriate values
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            size=text_size,
        )

    plt.tight_layout()
    # Save the figure if requested
    if savefig:
        plt.savefig("confusion_matrix.png")
    plt.show()
