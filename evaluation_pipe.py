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
        classification_evaluation_pipeline(y_true=y_test, y_pred=y_pred, y_prob=y_prob, classes=classes)
    """
    print("1. Printing Classification Report")
    print(classification_report(y_pred=y_pred, y_true=y_true, target_names=classes))
    print("2. Plot Confusion Matrix")
    make_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=classes)


def calculate_classes_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> pd.DataFrame:
    """
    Calculates precision, recall, and f1-score for each class based on the true and predicted labels.

    This function uses the `classification_report` from scikit-learn to generate a report on precision,
    recall, and f1-score for each class. It then organizes this information into a pandas DataFrame
    for easier analysis and visualization.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels, same shape as y_true.
        classes (np.ndarray): Array of class labels as strings.

    Returns:
        pd.DataFrame: A DataFrame containing the class names, their corresponding f1-score, precision,
                      and recall. Each row corresponds to a class.

    Example usage:
        class_names = ['cats', 'dogs']

        calculate_classes_metrics(y_true, y_pred, class_names=class_names)
    """
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )

    # Delete bottom Metrics
    report.pop("accuracy", None)
    report.pop("macro avg", None)
    report.pop("weighted avg", None)

    # generate pandas dataframe
    df = pd.DataFrame.from_dict(report).transpose().reset_index()
    df.rename(columns={"index": "class_name"}, inplace=True)

    return df


def plot_metric_from_classes(
    df: pd.DataFrame,
    metric: str,
    df_class_name_column: str = "class_name",
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Plots a horizontal bar chart of given metric scores for different classes.

    This function takes a pandas DataFrame containing metrics for different classes,
    a metric name to plot, and the DataFrame column name that contains class names.
    It then plots a horizontal bar chart showing the metric scores for each class,
    sorted in ascending order. Additionally, it annotates each bar with the metric score.

    Args:
        df (pd.DataFrame): The DataFrame containing the metric scores and class names.
        metric (str): The name of the metric column in `df` to plot.
                      This metric will be displayed on the x-axis. (precision, recall, f1-score or support)
        df_class_name_column (str): The name of the column in `df` that contains the class names.
                                    These class names will be displayed on the y-axis.
        figsize (tuple[int, int]): A tuple specifying the width and height in inches of the figure to be plotted.
                                   This allows customization of the plot size for better readability and fitting into different contexts.

    Returns:
        None: This function does not return a value. It generates a plot.

    Example usage:
        plot_metric_from_classes(df, metric='f1-score', df_class_name_column='class names', figsize=(10, 10))
    """
    fig, ax = plt.subplots(figsize=figsize)

    # sort df with ascending=True (necessary because ylabels wouldnt have the exact values)
    sorted_df = df.sort_values(by=[metric], ascending=True)

    # num_classes in range for y, x
    range_num_classes = range(len(sorted_df[df_class_name_column]))

    # create barh chart
    scores = ax.barh(range_num_classes, sorted_df[metric])
    ax.set_yticks(range_num_classes)
    ax.set_yticklabels(sorted_df[df_class_name_column])
    ax.set_xlabel(f"{metric}")
    ax.set_title(f"{metric} for Different Classes")

    # write to the right the metric score (%) for each class.
    for rect in scores:
        width = rect.get_width()
        ax.text(
            1.03 * width,
            rect.get_y() + rect.get_height() / 1.5,
            f"{width:.2f}",
            ha="center",
            va="center",
        )


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
