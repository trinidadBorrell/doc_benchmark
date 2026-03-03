"""MLP classification on pre-computed foundation model embeddings.

==================================================
MLP Embedding Classifier for VS vs MCS
==================================================

This script trains an MLP classifier on pre-computed embeddings from EEG
foundation models (CBraMod, TOTEM, LaBram) for binary classification of
consciousness states: VS (Vegetative State) vs MCS (Minimally Conscious State).

Key features:
- Binary classification: VS vs MCS (UWS -> VS, MCS+/MCS- -> MCS)
- Embedding-size agnostic: input dimension determined at runtime
- Cross-subject classification with GROUP-BASED splitting (NO data leakage)
- All sessions from the same subject stay together in train, val, OR test
- Two nested GroupShuffleSplits: outer for test, inner for val
- Balanced classes via pos_weight in BCEWithLogitsLoss
- Early stopping on validation loss
- Tracks loss and balanced error rate (1 - balanced accuracy) per epoch

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import numpy as np
import pandas as pd
import argparse
import os
import os.path as op
import json
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Prevent MKL/OpenBLAS segfaults when running as a subprocess
torch.set_num_threads(1)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score,
                             balanced_accuracy_score,
                             precision_recall_fscore_support, roc_curve, auc)

warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


# ======================================================================
# Model
# ======================================================================

class EmbeddingMLP(nn.Module):
    """Simple MLP for binary classification on embeddings.

    Architecture is embedding-size agnostic: input_dim is determined
    at runtime from the loaded data.
    """

    def __init__(self, input_dim):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input embeddings.
        """
        super().__init__()
        self.net = nn.Sequential(
         #   nn.Linear(input_dim, 128),
         #   nn.ReLU(),
         #   nn.BatchNorm1d(128),
         #   nn.Dropout(0.3),
         #   nn.Linear(128, 64),
         #   nn.ReLU(),
         #   nn.BatchNorm1d(64),
         #   nn.Dropout(0.3),
         #   nn.Linear(128, 1)
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ======================================================================
# Classifier
# ======================================================================

class EmbeddingMLPClassifier:
    """Cross-subject MLP classifier on foundation model embeddings.

    Uses two nested GroupShuffleSplits to create train / val / test sets
    with no subject overlap.  A single MLP is trained with early stopping
    on validation loss and evaluated on the held-out test set.
    """

    def __init__(self, data_dir, patient_labels_file, output_dir=None,
                 random_state=42, n_epochs=500, lr=1e-3, batch_size=32,
                 weight_decay=1e-4, patience=100):
        """
        Parameters
        ----------
        data_dir : str
            Path to directory containing sub-{ID}/ses-{NUM}/*_embedding.npy
        patient_labels_file : str
            Path to CSV with patient labels (must have diagnostic_crs_final
            column)
        output_dir : str
            Output directory for results
        random_state : int
            Random state for reproducibility
        n_epochs : int
            Maximum training epochs
        lr : float
            Learning rate
        batch_size : int
            Training batch size
        weight_decay : float
            L2 regularisation strength
        patience : int
            Early stopping patience (epochs without val loss improvement)
        """
        self.data_dir = data_dir
        self.patient_labels_file = patient_labels_file
        self.output_dir = output_dir or "results/mlp_embedding"
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience

        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Data containers
        self.X = None
        self.y = None
        self.subjects = []
        self.label_encoder = LabelEncoder()
        self.class_names = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_patient_labels(self):
        """Load patient labels for binary VS vs MCS classification.

        Uses the ``diagnostic_crs_final`` column, mapping UWS -> VS and
        MCS+/MCS- -> MCS.  All other states are skipped.

        Returns
        -------
        labels_dict : dict
            Mapping ``subject_session_key -> label``
        available_states : list
            Sorted list of unique states found
        """
        print(f"Loading patient labels from: {self.patient_labels_file}", flush=True)

        df = pd.read_csv(self.patient_labels_file)

        labels_dict = {}
        available_states = set()

        for _, row in df.iterrows():
            subject = row['subject']
            session = f"ses-{row['session']:02d}"
            state = row['diagnostic_crs_final']

            if pd.isna(state) or state == 'n/a':
                continue

            if state == 'UWS':
                state = 'VS'
            elif state in ['MCS+', 'MCS-']:
                state = 'MCS'
            else:
                continue

            key = f"{subject}_{session}"
            labels_dict[key] = state
            available_states.add(state)

        print(f"   Loaded labels for {len(labels_dict)} subject/sessions", flush=True)
        print(f"   Available states: {sorted(available_states)}", flush=True)
        return labels_dict, sorted(available_states)

    def load_embeddings(self, embedding_suffix="_embedding.npy"):
        """Discover and load embedding files, mean-pooling across windows.

        Parameters
        ----------
        embedding_suffix : str
            File suffix to match (default: ``_embedding.npy``).

        Returns
        -------
        dict
            Mapping ``subject_session_key -> (embedding_dim,)`` numpy array.
        """
        print(f"Loading embeddings from: {self.data_dir}", flush=True)
        embeddings = {}

        subject_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if d.startswith('sub-') and op.isdir(op.join(self.data_dir, d))
        ])

        for subject_dir in subject_dirs:
            subject_id = subject_dir.replace('sub-', '')
            subject_path = op.join(self.data_dir, subject_dir)

            session_dirs = sorted([
                d for d in os.listdir(subject_path)
                if d.startswith('ses-') and op.isdir(op.join(subject_path, d))
            ])

            for session_dir in session_dirs:
                session_path = op.join(subject_path, session_dir)

                emb_files = sorted([
                    f for f in os.listdir(session_path)
                    if (f.endswith(embedding_suffix)
                        or f.endswith("_embeddings.npy"))
                    and not f.endswith("_metadata.npy")
                ])

                if not emb_files:
                    continue

                emb_path = op.join(session_path, emb_files[0])
                try:
                    data = np.load(emb_path)
                    original_shape = data.shape

                    # Reduce to 1-D embedding by averaging over
                    # spatial / temporal dimensions
                    if data.ndim == 1:
                        pass  # already 1-D
                    elif data.ndim == 2:
                        # (n_windows, emb_dim) -> (emb_dim,)
                        data = data.mean(axis=0)
                    elif data.ndim >= 3:
                        # e.g. (n_windows, n_channels, ..., emb_dim)
                        axes_to_average = tuple(range(data.ndim - 1))
                        data = data.mean(axis=axes_to_average)
                        print(f"   Pooled {original_shape} -> {data.shape} "
                              f"for {op.basename(emb_path)}", flush=True)
                    else:
                        print(f"   Unexpected shape {data.shape} in "
                              f"{emb_path}, skipping", flush=True)
                        continue

                    key = f"{subject_id}_{session_dir}"
                    embeddings[key] = data
                except Exception as e:
                    print(f"   Error loading {emb_path}: {e}", flush=True)

        print(f"   Loaded embeddings for {len(embeddings)} subject/sessions", flush=True)
        return embeddings

    def collect_data(self, embedding_suffix="_embedding.npy"):
        """Match embeddings with labels and build feature / label arrays.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y_encoded : np.ndarray, shape (n_samples,)
        subjects : list of str
        """
        print("Collecting data ...", flush=True)

        labels_dict, _ = self.load_patient_labels()
        embeddings = self.load_embeddings(embedding_suffix)

        X_list, y_list, subjects_list = [], [], []

        for key, emb in sorted(embeddings.items()):
            if key not in labels_dict:
                continue

            if X_list and emb.shape != X_list[0].shape:
                print(f"   Shape mismatch for {key}: expected "
                      f"{X_list[0].shape}, got {emb.shape}", flush=True)
                continue

            X_list.append(emb)
            y_list.append(labels_dict[key])
            subjects_list.append(key)

        if not X_list:
            raise ValueError("No subjects matched between embeddings "
                             "and labels!")

        self.X = np.array(X_list)
        self.y = np.array(y_list)
        self.subjects = subjects_list

        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = self.label_encoder.classes_

        print(f"   Dataset: {self.X.shape[0]} samples x "
              f"{self.X.shape[1]} features", flush=True)
        print(f"   Classes: {list(self.class_names)}", flush=True)
        unique, counts = np.unique(self.y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"      {cls}: {cnt}", flush=True)

        return self.X, self.y_encoded, self.subjects

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _make_loader(self, X, y, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _train_model(self, X_train, y_train, X_val, y_val, pos_weight):
        """Train an MLP with early stopping on validation loss.

        Additionally tracks balanced error rate (1 - balanced accuracy)
        on both train and val sets every epoch.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data and encoded labels.
        X_val, y_val : np.ndarray
            Validation data and encoded labels.
        pos_weight : float
            Weight for the positive class in BCEWithLogitsLoss.

        Returns
        -------
        model : EmbeddingMLP
            Best model (lowest val loss).
        train_losses : list of float
        val_losses : list of float
        train_errors : list of float
            1 - balanced_accuracy on train set per epoch.
        val_errors : list of float
            1 - balanced_accuracy on val set per epoch.
        """
        input_dim = X_train.shape[1]
        model = EmbeddingMLP(input_dim).to(self.device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )
        optimizer = optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        train_losses, val_losses = [], []
        train_errors, val_errors = [], []

        for epoch in range(self.n_epochs):
            # --- train ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            all_train_preds, all_train_labels = [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                all_train_preds.append(preds)
                all_train_labels.append(yb.cpu().numpy().astype(int))

            train_losses.append(epoch_loss / max(n_batches, 1))
            train_preds = np.concatenate(all_train_preds)
            train_labels = np.concatenate(all_train_labels)
            train_errors.append(
                1.0 - balanced_accuracy_score(train_labels, train_preds))

            # --- validate ---
            model.eval()
            val_loss = 0.0
            n_val = 0
            all_val_preds, all_val_labels = [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()
                    n_val += 1

                    preds = (
                        (torch.sigmoid(logits) >= 0.5).long().cpu().numpy())
                    all_val_preds.append(preds)
                    all_val_labels.append(yb.cpu().numpy().astype(int))

            val_losses.append(val_loss / max(n_val, 1))
            val_preds = np.concatenate(all_val_preds)
            val_labels = np.concatenate(all_val_labels)
            val_errors.append(
                1.0 - balanced_accuracy_score(val_labels, val_preds))

            # --- epoch log ---
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      Epoch {epoch + 1:>3d}/{self.n_epochs}  "
                      f"train_loss={train_losses[-1]:.4f}  "
                      f"val_loss={val_losses[-1]:.4f}  "
                      f"train_err={train_errors[-1]:.4f}  "
                      f"val_err={val_errors[-1]:.4f}", flush=True)

            # --- early stopping ---
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, train_losses, val_losses, train_errors, val_errors

    @torch.no_grad()
    def _predict(self, model, X):
        """Return (probabilities, predicted_labels) for given data."""
        model.eval()
        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)
        all_probs = []
        for xb, _ in loader:
            xb = xb.to(self.device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
        probs = np.concatenate(all_probs)
        preds = (probs >= 0.5).astype(int)
        return probs, preds

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_classification(self, test_size=0.2, val_size=0.2):
        """Run the full classification pipeline.

        Steps
        -----
        1. Collect data (embeddings + labels)
        2. Outer split: GroupShuffleSplit -> trainval / test
        3. Inner split: GroupShuffleSplit -> train / val
        4. Train MLP with early stopping on val loss
        5. Evaluate on held-out test set
        6. Save results + plots

        Parameters
        ----------
        test_size : float
            Fraction of subjects held out for testing.
        val_size : float
            Fraction of remaining subjects used for validation.
        """
        print("=" * 80, flush=True)
        print("MLP EMBEDDING CLASSIFICATION (VS vs MCS)", flush=True)
        print("=" * 80, flush=True)
        print(f"Data directory: {self.data_dir}", flush=True)
        print(f"Labels file: {self.patient_labels_file}", flush=True)
        print(f"Output directory: {self.output_dir}", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Epochs: {self.n_epochs}, LR: {self.lr}, "
              f"Batch: {self.batch_size}", flush=True)
        print(flush=True)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 1. Collect data
        X, y, subjects = self.collect_data()

        # 2. Subject groups (prevent leakage)
        groups = np.array([s.split('_ses-')[0] for s in subjects])
        unique_groups = np.unique(groups)
        print(f"   {len(unique_groups)} unique subjects across "
              f"{len(subjects)} sessions", flush=True)

        # 3. Outer split: trainval / test
        gss_outer = GroupShuffleSplit(
            n_splits=1, test_size=test_size,
            random_state=self.random_state)
        trainval_idx, test_idx = next(gss_outer.split(X, y, groups=groups))

        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]
        groups_trainval = groups[trainval_idx]
        subjects_test = [subjects[i] for i in test_idx]

        # 4. Inner split: train / val
        gss_inner = GroupShuffleSplit(
            n_splits=1, test_size=val_size,
            random_state=self.random_state)
        train_idx_inner, val_idx_inner = next(
            gss_inner.split(X_trainval, y_trainval, groups=groups_trainval))

        X_train = X_trainval[train_idx_inner]
        y_train = y_trainval[train_idx_inner]
        X_val = X_trainval[val_idx_inner]
        y_val = y_trainval[val_idx_inner]
        groups_train = groups_trainval[train_idx_inner]
        groups_val = groups_trainval[val_idx_inner]

        subjects_train = [subjects[trainval_idx[i]]
                          for i in train_idx_inner]
        subjects_val = [subjects[trainval_idx[i]]
                        for i in val_idx_inner]

        # Verify no subject leakage across any pair of splits
        train_groups_set = set(groups_train)
        val_groups_set = set(groups_val)
        test_groups_set = set(groups[test_idx])

        if train_groups_set & val_groups_set:
            raise ValueError(f"Train/val leakage: "
                             f"{train_groups_set & val_groups_set}")
        if train_groups_set & test_groups_set:
            raise ValueError(f"Train/test leakage: "
                             f"{train_groups_set & test_groups_set}")
        if val_groups_set & test_groups_set:
            raise ValueError(f"Val/test leakage: "
                             f"{val_groups_set & test_groups_set}")

        print(f"   Train: {len(X_train)} sessions "
              f"({len(train_groups_set)} subjects)", flush=True)
        print(f"   Val:   {len(X_val)} sessions "
              f"({len(val_groups_set)} subjects)", flush=True)
        print(f"   Test:  {len(X_test)} sessions "
              f"({len(test_groups_set)} subjects)", flush=True)

        # Class balance info
        for name, y_split in [("Train", y_train), ("Val", y_val),
                               ("Test", y_test)]:
            u, c = np.unique(y_split, return_counts=True)
            dist = ", ".join(f"{self.class_names[ci]}: {n}"
                             for ci, n in zip(u, c))
            print(f"   {name} class distribution: {dist}", flush=True)

        # pos_weight for BCEWithLogitsLoss (from training set only)
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_weight = n_neg / max(n_pos, 1)
        print(f"   pos_weight (class balance): {pos_weight:.3f}", flush=True)

        # 5. Train MLP with early stopping
        print(f"\n   Training MLP (max {self.n_epochs} epochs, "
              f"patience {self.patience}) ...", flush=True)
        model, train_losses, val_losses, train_errors, val_errors = \
            self._train_model(X_train, y_train, X_val, y_val, pos_weight)
        n_trained_epochs = len(train_losses)
        print(f"   Training stopped after {n_trained_epochs} epochs", flush=True)

        # Val set performance (at best checkpoint)
        val_probs, val_preds = self._predict(model, X_val)
        val_bal_acc = balanced_accuracy_score(y_val, val_preds)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"   Val balanced accuracy: {val_bal_acc:.3f}", flush=True)

        # 6. Evaluate on test set
        print("   Evaluating on held-out test set ...", flush=True)
        test_probs, test_preds = self._predict(model, X_test)

        test_bal_acc = balanced_accuracy_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, test_preds)

        test_auc = None
        if len(np.unique(y_test)) == 2:
            try:
                test_auc = roc_auc_score(y_test, test_probs)
            except Exception:
                pass

        test_precision, test_recall, test_f1, test_support = \
            precision_recall_fscore_support(y_test, test_preds, average=None,
                                            zero_division=0)
        test_conf = confusion_matrix(y_test, test_preds)
        test_report = classification_report(y_test, test_preds,
                                             output_dict=True,
                                             zero_division=0)

        results = {
            'val_balanced_accuracy': float(val_bal_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'test_balanced_accuracy': float(test_bal_acc),
            'test_auc_score': (float(test_auc)
                               if test_auc is not None else None),
            'test_precision': test_precision.tolist(),
            'test_recall': test_recall.tolist(),
            'test_f1_score': test_f1.tolist(),
            'test_support': test_support.tolist(),
            'test_confusion_matrix': test_conf.tolist(),
            'test_classification_report': test_report,
            'class_names': list(self.class_names),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': int(X.shape[1]),
            'training_epochs': n_trained_epochs,
            'subjects_train': subjects_train,
            'subjects_val': subjects_val,
            'subjects_test': subjects_test,
            'y_test_true': y_test.tolist(),
            'y_test_pred': test_preds.tolist(),
            'y_test_probs': test_probs.tolist(),
        }

        # 7. Save
        self._save_results(results, model)
        self._plot_results(results, train_losses, val_losses,
                           train_errors, val_errors)

        # Summary
        print("\n" + "=" * 80, flush=True)
        print("MLP EMBEDDING CLASSIFICATION SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"   Trained for {n_trained_epochs} epochs", flush=True)
        print(f"   Val balanced accuracy:  {val_bal_acc:.3f}", flush=True)
        print(f"   Test balanced accuracy: {test_bal_acc:.3f}", flush=True)
        if test_auc is not None:
            print(f"   Test AUC-ROC: {test_auc:.3f}", flush=True)
        print(f"   Results saved to: {self.output_dir}", flush=True)
        print("=" * 80, flush=True)

        return results

    # ------------------------------------------------------------------
    # Saving & plotting
    # ------------------------------------------------------------------

    def _save_results(self, results, model):
        """Save JSON results, trained model weights, and predictions CSV."""
        results_file = op.join(self.output_dir, 'classification_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved to: {results_file}", flush=True)

        model_file = op.join(self.output_dir, 'trained_model.pt')
        torch.save(model.state_dict(), model_file)

        pred_labels = [self.class_names[p] for p in results['y_test_pred']]
        true_labels = [self.class_names[t] for t in results['y_test_true']]
        df = pd.DataFrame({
            'subject_session': results['subjects_test'],
            'true_state': true_labels,
            'predicted_state': pred_labels,
            'correct': [t == p for t, p in zip(true_labels, pred_labels)],
            'prob_positive': results['y_test_probs'],
        })
        csv_file = op.join(self.output_dir, 'subject_predictions.csv')
        df.to_csv(csv_file, index=False)

    def _plot_results(self, results, train_losses, val_losses,
                      train_errors, val_errors):
        """Generate a 2x2 figure: loss, confusion matrix, error, ROC."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(
            'MLP Embedding Classification Results\n'
            f'Train: {results["n_train"]}, '
            f'Val: {results["n_val"]}, '
            f'Test: {results["n_test"]}',
            fontsize=16)

        epochs_x = np.arange(len(train_losses))

        # --- (0, 0) Training & validation loss ---
        ax = axes[0, 0]
        ax.plot(epochs_x, train_losses, color='tab:blue', lw=2,
                label='Train')
        ax.plot(epochs_x, val_losses, color='tab:orange', lw=2,
                label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (BCEWithLogitsLoss)')
        ax.set_title('Training & Validation Loss')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # --- (0, 1) Confusion matrix ---
        ax = axes[0, 1]
        conf = np.array(results['test_confusion_matrix'])
        im = ax.imshow(conf, cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        thresh = conf.max() / 2.0
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                ax.text(j, i, str(conf[i, j]), ha='center', va='center',
                        color='white' if conf[i, j] > thresh else 'black',
                        fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names)
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (Test Set)')

        # --- (1, 0) Balanced error rate ---
        ax = axes[1, 0]
        ax.plot(epochs_x, train_errors, color='tab:blue', lw=2,
                label='Train')
        ax.plot(epochs_x, val_errors, color='tab:orange', lw=2,
                label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Balanced Error Rate (1 - Bal. Acc.)')
        ax.set_title('Training & Validation Error')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # --- (1, 1) ROC curve ---
        ax = axes[1, 1]
        if results['test_auc_score'] is not None:
            y_true = np.array(results['y_test_true'])
            y_probs = np.array(results['y_test_probs'])
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'navy', lw=2, ls='--')
            ax.legend(loc='lower right')
        else:
            ax.text(0.5, 0.5, 'AUC not available', ha='center',
                    va='center', transform=ax.transAxes)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Test Set)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = op.join(self.output_dir, 'classification_results.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Plot saved to: {plot_file}", flush=True)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MLP classification on pre-computed EEG foundation '
                    'model embeddings')

    parser.add_argument('--data-dir', required=True,
                        help='Path to directory with '
                             'sub-{ID}/ses-{NUM}/*_embedding.npy')
    parser.add_argument('--patient-labels', required=True,
                        help='Path to patient_labels_with_controls.csv')
    parser.add_argument('--output-dir',
                        help='Output directory for results')
    parser.add_argument('--n-epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of subjects for test set')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Fraction of remaining subjects for '
                             'validation set')
    parser.add_argument('--random-state', type=int, default=42)

    args = parser.parse_args()

    classifier = EmbeddingMLPClassifier(
        data_dir=args.data_dir,
        patient_labels_file=args.patient_labels,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    classifier.run_classification(
        test_size=args.test_size,
        val_size=args.val_size,
    )


if __name__ == '__main__':
    main()
