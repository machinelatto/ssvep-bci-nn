from matplotlib import pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns


def split_train_test_trials(X, y, test_trial):
    y_test = []
    y_train = []
    X_test = []
    X_train = []
    for trial in range(0, 6):
        if trial == test_trial:
            X_test.extend(X[trial])
            y_test.extend(y[trial])
        else:
            X_train.extend(X[trial])
            y_train.extend(y[trial])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def prepare_data_trials(
    X, y, test_trial=5, train_batch_size=16, test_batch_size=1, device=0
):
    print(f"Testing on trial {test_trial}.")
    X_train, X_test, y_train, y_test = split_train_test_trials(X, y, test_trial)
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.long).to(device),
    )
    test_data = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.long).to(device),
    )
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader


def split_train_test_subjects(X, y, test_subject):
    y_test = []
    y_train = []
    X_test = []
    X_train = []
    for subject in range(0, X.shape[0]):
        if subject == test_subject:
            X_test.extend(X[subject])
            y_test.extend(y[subject])
        else:
            X_train.extend(X[subject])
            y_train.extend(y[subject])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def prepare_data_subjects(
    X,
    y,
    test_subject=0,
    train_batch_size=16,
    test_batch_size=1,
    device=0,
    validation_split=False,
):
    print(f"Testing on subject {test_subject + 1}.")
    X_train_full, X_test, y_train_full, y_test = split_train_test_subjects(
        X, y, test_subject
    )

    if validation_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )
        train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).to(device),
            torch.tensor(y_train, dtype=torch.long).to(device),
        )
        test_data = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32).to(device),
            torch.tensor(y_test, dtype=torch.long).to(device),
        )
        val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32).to(device),
            torch.tensor(y_val, dtype=torch.long).to(device),
        )
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=True)
        print(f"Training dataset size: {len(y_train)}")
        print(f"Validation dataset size: {len(y_val)}")
        print(f"Test dataset size: {len(y_test)}")
        return train_loader, test_loader, val_loader

    else:
        train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).to(device),
            torch.tensor(y_train, dtype=torch.long).to(device),
        )
        test_data = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32).to(device),
            torch.tensor(y_test, dtype=torch.long).to(device),
        )
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
        print(f"Training dataset size: {len(y_train)}")
        print(f"Test dataset size: {len(y_test)}")
        return train_loader, test_loader


def get_desired_freqs_and_classes(
    full_signal, all_freqs, desired_channels, desired_freqs, number_of_windows=5
):
    labels = []
    signals = []
    for trial in range(0, 6):
        labels_trial = []
        signals_trial = []
        for idx, freq in enumerate(desired_freqs):
            for window in range(0, number_of_windows):
                if number_of_windows > 1:
                    signal = full_signal[
                        desired_channels,
                        :,
                        np.where(all_freqs == freq)[1],
                        trial,
                        window,
                    ]
                else:
                    signal = full_signal[
                        desired_channels, :, np.where(all_freqs == freq)[1], trial
                    ]
                signals_trial.append(signal)
                labels_trial.append(idx)
        labels.append(labels_trial)
        signals.append(signals_trial)

    signals = np.array(signals)
    labels = np.array(labels)
    print(signals.shape)
    print(labels.shape)
    return signals, labels


def plot_confusion_matrix(all_labels, all_preds, class_labels, filename=None):
    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Normalize the confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # Create a figure and increase the figure size
    plt.figure(figsize=(12, 10))
    # Create a heatmap using seaborn
    sns.heatmap(
        cm_normalized,
        vmin=0,
        vmax=1,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        square=True,
    )
    # Set labels and title
    plt.xlabel("Predicted", fontweight="bold")
    plt.ylabel("True", fontweight="bold")
    plt.title("Normalized Confusion Matrix", fontsize=16, fontweight="bold")
    # Increase font size of annotations
    # plt.tick_params(labelsize=12)
    # Adjust layout and display the plot
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def plot_reordered_confusion_matrix(all_labels, all_preds, class_labels, filename=None):
    # Convert class_labels to numpy array if it's not already
    classes = np.arange(len(class_labels))  # Sort class_labels in ascending order
    sorted_indices = np.argsort(class_labels)
    classes_sorted = classes[sorted_indices]
    class_labels_sorted = class_labels[sorted_indices]

    # Compute the confusion matrix with sorted labels
    cm = confusion_matrix(all_labels, all_preds, labels=classes_sorted)

    # Normalize the confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure and increase the figure size
    plt.figure(figsize=(20, 16))
    # Create a heatmap using seaborn
    sns.heatmap(
        cm_normalized,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels_sorted,
        yticklabels=class_labels_sorted,
        square=True,
    )

    # Set labels and title
    plt.xlabel("Predicted", fontweight="bold")
    plt.ylabel("True", fontweight="bold")
    plt.title("Normalized Confusion Matrix (Sorted)", fontsize=16, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Adjust layout and display the plot
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()


def evaluate(
    model, test_loader, class_labels, device=0, filename=None, one_channel=True
):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if one_channel:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)

            # Predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    plot_reordered_confusion_matrix(all_labels, all_preds, class_labels, filename)
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Test set Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return accuracy, recall, f1
