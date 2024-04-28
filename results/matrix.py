import matplotlib.pyplot as plt
import seaborn as sns

# f1_scores_categories = {
#     "True Offensive": 0.6114,
#     "True Neither": 0.7601
# }

before_f1_scores_categories = {
    "True Offensive": 0.5693,
    "True Neither": 0.6627
}

def plot_confusion_matrix(f1_scores):
    """
    Plot the confusion matrix using F1 scores for true and predicted categories.

    Args:
    f1_scores (dict): A dictionary containing the F1 scores for each category.

    Returns:
    None
    """
    true_offensive_f1 = f1_scores["True Offensive"]
    true_neither_f1 = f1_scores["True Neither"]

    confusion_matrix = [[true_offensive_f1, 1 - true_offensive_f1],
                        [1 - true_neither_f1, true_neither_f1]]

    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=["Offensive", "Neither"], yticklabels=["Offensive", "Neither"])

    plt.xlabel("Predicted Categories")
    plt.ylabel("True Categories")
    plt.title("Confusion Matrix (F1 Scores)")

    # Save the plot as a PNG file
    plt.savefig('f1_scores_matrix.png')

    plt.show()


plot_confusion_matrix(before_f1_scores_categories)
