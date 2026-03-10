import os
import pandas as pd
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# CONFIGURATION (Focus on Generalization & 90%+ Hamming Acc)
# ==========================================
MODEL_NAME = "google/muril-base-cased"
OUTPUT_DIR = "saved_model"
TRAIN_FILE = "train.csv"
VALID_FILE = "validate.csv"

# --- Hyperparameters ---
EPOCHS = 12               # Reduced to prevent overfitting
BATCH_SIZE = 16
LEARNING_RATE = 1e-5      # Lower LR for more stable learning
MAX_LENGTH = 128
WEIGHT_DECAY = 0.05       # Increased weight decay (regularization)
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 4

LABELS = ["hate", "offensive", "defamation", "fake", "non-hostile"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        with torch.no_grad():
            targets_smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
            
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets_smoothed, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)
        return loss.mean()

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Post', 'Labels Set'])
    df['label_list'] = df['Labels Set'].apply(lambda x: [label.strip() for label in x.split(',')])
    def to_multi_hot(label_list):
        vector = [0.0] * len(LABELS)
        for label in label_list:
            if label in LABELS: vector[LABELS.index(label)] = 1.0
        return vector
    df['labels'] = df['label_list'].apply(to_multi_hot)
    return df[['Post', 'labels']]

print("Loading datasets...")
train_df = load_and_preprocess(TRAIN_FILE)
valid_df = load_and_preprocess(VALID_FILE)

# --- Moderate Oversampling ---
print("  Applying moderate oversampling...")
train_labels_array = np.array(train_df['labels'].tolist())
minority_labels = ['hate', 'offensive', 'defamation']
oversampled_dfs = [train_df]
for label_name in minority_labels:
    l_idx = LABELS.index(label_name)
    mask = train_labels_array[:, l_idx] == 1
    minority_samples = train_df[mask]
    target = 1200 # Lower target to prevent memorization
    if len(minority_samples) < target:
        oversampled_dfs.append(minority_samples.sample(n=target-len(minority_samples), replace=True, random_state=42))

train_df_oversampled = pd.concat(oversampled_dfs, ignore_index=True).sample(frac=1, random_state=42)

# Compute weights
label_counts = np.array(train_df_oversampled['labels'].tolist()).sum(axis=0)
class_weights = len(train_df_oversampled) / (len(LABELS) * label_counts + 1e-6)
class_weights = torch.tensor(class_weights / class_weights.sum() * len(LABELS), dtype=torch.float).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def create_dataloader(df, batch_size, shuffle=True):
    tokens = tokenizer(df['Post'].tolist(), padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    labels = torch.tensor(df['labels'].tolist(), dtype=torch.float)
    return DataLoader(TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels), batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(train_df_oversampled, BATCH_SIZE)
valid_loader = create_dataloader(valid_df, BATCH_SIZE, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*EPOCHS*WARMUP_RATIO), num_training_steps=len(train_loader)*EPOCHS)
loss_fn = FocalLossWithSmoothing(alpha=class_weights, smoothing=0.1)

print("\nStarting specialized training loop...")
best_val_loss = float('inf')
best_model_state = None
patience = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        loss = loss_fn(model(b_input_ids, attention_mask=b_mask).logits, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    for batch in valid_loader:
        b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_mask).logits
            val_loss += loss_fn(logits, b_labels).item()
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(valid_loader)
    all_preds_arr = (np.array(all_preds) > 0.5).astype(int)
    all_labels_arr = np.array(all_labels)
    h_acc = 1 - hamming_loss(all_labels_arr, all_preds_arr)
    em_acc = accuracy_score(all_labels_arr, all_preds_arr)
    f1 = f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
    
    print(f"Epoch {epoch+1}: ValLoss={avg_val_loss:.4f}, HammingAcc={h_acc:.4f}, ExactMatch={em_acc:.4f}, F1={f1:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience = 0
        print("  ★ New best!")
    else:
        patience += 1
        if patience >= EARLY_STOPPING_PATIENCE: break

model.load_state_dict(best_model_state)
print("\nSaving final model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Optimal thresholds logic
model.eval()
all_probs = []
all_labels = []
for batch in valid_loader:
    b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
    with torch.no_grad():
        all_probs.extend(torch.sigmoid(model(b_input_ids, attention_mask=b_mask).logits).cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
thresholds = {}
for i, label in enumerate(LABELS):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.2, 0.8, 0.05):
        f1 = f1_score(all_labels[:, i], (all_probs[:, i] > t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    thresholds[label] = float(best_t)
    print(f"  {label}: {best_t:.2f}")

with open(os.path.join(OUTPUT_DIR, "optimal_thresholds.json"), "w") as f:
    import json
    json.dump(thresholds, f, indent=2)
print("Done!")
