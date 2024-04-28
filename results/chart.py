import matplotlib.pyplot as plt
import seaborn as sns


# F1 scores before and after fine-tuning
classes = ['Offensive', 'Overall']


def plot_comparison_chart():
    # f1_scores_before = [0.9282, 0.8155, 0.9004]
    # f1_scores_after = [0.9268, 0.8037, 0.8954]

    f1_scores_before = [0.6061, 0.6061]
    f1_scores_after = [0.5443, 0.5443]

    # Plotting F1 scores before and after fine-tuning
    x = range(len(classes))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x, f1_scores_before, width, label='Before Fine-tuning')
    bars2 = ax.bar([p + width for p in x], f1_scores_after, width, label='After Fine-tuning')

    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Sample 3: F1 Scores Before and After Fine-tuning')
    ax.set_xticks([p + 0.5 * width for p in x])
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('f1_scores_comparison.png')


plot_comparison_chart()
