import csv

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# Load the saved model
model_path = "roberta_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using GPU: {torch.cuda.is_available()}")


model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# Function to classify tweets
def classify_text(tweet):
    inputs = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=500,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )
    # Move input tensors to the appropriate device (GPU)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class, probabilities


class_map = {0: 'Neither', 1: 'Offensive'}


def calculate_f1_scores(file_path, dataset='test'):
    """
    Calculate F1 scores based on predictions.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    None
    """
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            try:
                class_label = int(row['class'])
                tweet = row['text']

                # Assuming you have a function classify_tweet that returns predicted class
                predicted_class, _ = classify_text(tweet)

                y_true.append(class_label)
                y_pred.append(predicted_class)
            except (UnicodeDecodeError, ValueError) as e:
                pass

    # Calculate F1 scores
    f1_scores = f1_score(y_true, y_pred, average=None)

    # Print the F1 scores
    print(f"{dataset} F1 Scores for each class:")
    for class_label, f1_score_val in enumerate(f1_scores):
        print(f"{class_map.get(class_label)}: {f1_score_val}")

    # Calculate overall F1 score before merging
    overall_f1_score_before_merge = f1_score(y_true, y_pred, average='weighted')
    print(f"Overall F1 Score: {overall_f1_score_before_merge}")

def calculate_f1_scores_categories(file_path, dataset='test'):
    """
    Calculate F1 scores for True Offensive, True Neither, Predicted Offensive, and Predicted Neither categories.

    Args:
    file_path (str): The path to the CSV file.
    dataset (str): The name of the dataset.

    Returns:
    None
    """
    true_offensive = []
    true_neither = []
    pred_offensive = []
    pred_neither = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            try:
                class_label = int(row['class'])
                tweet = row['text']

                # Assuming you have a function classify_tweet that returns predicted class
                predicted_class, _ = classify_text(tweet)

                if class_label == 1:  # Offensive
                    true_offensive.append(class_label)
                    pred_offensive.append(predicted_class)
                else:  # Neither
                    true_neither.append(class_label)
                    pred_neither.append(predicted_class)
            except (UnicodeDecodeError, ValueError) as e:
                pass

    # Calculate F1 scores
    f1_score_offensive = f1_score(true_offensive, pred_offensive)
    f1_score_neither = f1_score(true_neither, pred_neither)

    # Print the F1 scores
    print(f"{dataset} F1 Scores:")
    print("True Offensive:", f1_score_offensive)
    print("True Neither:", f1_score_neither)


def remove_non_utf8_chars(input_string):
    return input_string.encode('utf-8', 'ignore').decode('utf-8')


def save_output_file(output_file_path, content):
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        output_file.write(content)


if __name__ == "__main__":
    # calculate_f1_scores('sample1.csv', "sample 1")
    # print("----------------------------------")
    calculate_f1_scores('samples/sample3.csv', "sample 1")
    # calculate_f1_scores_categories('samples/sample1.csv', "sample 2")
    # print("----------------------------------")
    # calculate_f1_scores('samples/sample2.csv', "sample 2")
    # print("----------------------------------")
    # calculate_f1_scores('samples/sample3.csv', "sample 3")
