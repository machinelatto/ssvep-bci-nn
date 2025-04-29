from models.DNN import SSVEPDNN

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm

from utils.utils import evaluate, prepare_data_trials
from pathlib import Path
import pandas as pd

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    device=0,
    save_path="best_model_raw.pth",
):
    best_val_accuracy = 0.0
    model.to(device)
    early_stop_count = 0
    current_min_val_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # eval train
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        avg_train_loss = running_loss / len(train_loader)

        # eval validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # val accuracy
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Save if best vall acc
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_val_accuracy:.4f}")

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )
        if (avg_val_loss + 0.0001) <= current_min_val_loss:
            current_min_val_loss = avg_val_loss
            early_stop_count = 0
            print("Validation loss decreased...")
        else:
            early_stop_count += 1
            if early_stop_count >= 1000:
                print(
                    "Validation loss has not decreased for 1000 epochs. Stoping training..."
                )
                break


def run_and_evaluate_subject(
    subject, signals, labels, batch_size, epochs, freqs_labels
):
    test_subject = subject

    train_loader, test_loader = prepare_data_trials(
        signals,
        labels,
        test_trial=5,
        train_batch_size=batch_size,
    )
    validation_loader = test_loader
    n_classes = len(freqs_labels)
    model = SSVEPDNN(n_classes, 9, 250, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.001, lr=0.0001)

    # Train the model
    best_model_save_path = RESULTS_DIR.joinpath(f"{test_subject + 1}_best_model.pth")

    train(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        num_epochs=epochs,
        save_path=best_model_save_path,
    )

    # Eval
    best_model = SSVEPDNN(n_classes, 9, 250, 3)
    best_model.load_state_dict(torch.load(best_model_save_path))
    best_model.to(device)
    cm_path = RESULTS_DIR.joinpath(f"{test_subject + 1}_cm.png")
    acc, rcll, f1s = evaluate(
        best_model,
        test_loader,
        class_labels=freqs_labels,
        filename=cm_path,
        one_channel=False,
    )
    metrics = {
        "accuracy": np.array(acc),
        "recall": np.array(rcll),
        "f1_score": np.array(f1s),
    }

    return metrics


def run_model_for_all_subjects(
    data_path, labels_path, batch_size, channels, classes, epochs
):
    results = {}
    processed_signals = np.load(data_path)
    labels = np.load(labels_path)

    for subject in range(processed_signals.shape[0]):
        print(f"Running for subject {subject+1} ---------------")
        print(processed_signals[subject].shape)
        print(labels[subject])
        results[subject] = run_and_evaluate_subject(
            subject,
            processed_signals[subject],
            labels[subject],
            batch_size,
            epochs,
            classes,
        )
    return results


if __name__ == "__main__":
    RESULTS_DIR = Path("DNN_intra/40freq/test_trial_5")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # DATASET = Path("../processed_datasets/sinais_filtrados_6_70_Hz_janelas_1s/")
    DATASET = Path("pre_processed_3_subbands_40_classesTrue.npy")
    LABELS = Path("labels_40_classesTrue.npy")

    selected_freqs = np.array(
        [np.round(i, 2) for i in np.linspace(8, 15.8, 40)]
    )  # Frequências de interesse
    # selected_freqs = np.array([i for i in range(8, 16)])
    # selected_freqs = np.array([8.2, 10.8, 12.6, 15.4])
    channels = [
        47,
        53,
        54,
        55,
        56,
        57,
        60,
        61,
        62,
    ]  # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2

    epochs = 1000

    results_dict = run_model_for_all_subjects(
        DATASET, LABELS, 100, channels, selected_freqs, epochs
    )
    with open(RESULTS_DIR.joinpath("results_dict.pkl"), "wb") as f:
        pickle.dump(results_dict, f)

    df = pd.DataFrame.from_dict(results_dict, orient="index")

    df.to_csv(RESULTS_DIR.joinpath("results.csv"))
