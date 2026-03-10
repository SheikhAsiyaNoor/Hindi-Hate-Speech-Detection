import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use the saved model directory if it exists
MODEL_DIR = "saved_model" if os.path.exists("saved_model") else "indicbert_model"
LABELS = ["hate", "offensive", "defamation", "fake", "non-hostile"]
DEFAULT_THRESHOLD = 0.5

def main():
    print("-" * 50)
    print(f"Loading model and tokenizer from '{MODEL_DIR}'...")
    
    # Load optimal thresholds if available
    thresholds_path = os.path.join(MODEL_DIR, "optimal_thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            thresholds_dict = json.load(f)
        thresholds = [thresholds_dict.get(label, DEFAULT_THRESHOLD) for label in LABELS]
        print(f"Loaded optimal thresholds from {thresholds_path}")
    else:
        thresholds = [DEFAULT_THRESHOLD] * len(LABELS)
        print(f"Warning: optimal_thresholds.json not found. Using default {DEFAULT_THRESHOLD}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, 
            num_labels=len(LABELS), 
            problem_type="multi_label_classification"
        )
        model.eval()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    print("-" * 50)
    print("Hindi Hate Speech Detection - Testing CLI")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 50)

    while True:
        try:
            text = input("\nEnter Hindi text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
            
        if text.lower() in ['quit', 'exit']:
            print("Exiting...")
            break
            
        if not text:
            print("Please enter some text.")
            continue

        # Tokenize input (using max_length=128 to match training)
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).squeeze().tolist()
            
        if not isinstance(probabilities, list):
            probabilities = [probabilities]

        triggered_labels = []
        print("\nPredictions:")
        for label, prob, thresh in zip(LABELS, probabilities, thresholds):
            is_triggered = prob > thresh
            if is_triggered:
                triggered_labels.append(label)
            # Use specific threshold for each label
            marker = "-->" if is_triggered else "   "
            print(f"{marker} {label:<12} : {prob:.4f} ({prob*100:.2f}%) [threshold: {thresh:.2f}]")

        # FALLBACK LOGIC: If no hostile labels triggered, it's non-hostile
        # This fixes the "suspicious" false positives on clean sentences
        hostile_labels = ["hate", "offensive", "defamation", "fake"]
        any_hostile_triggered = any(label in triggered_labels for label in hostile_labels)
        
        if not any_hostile_triggered:
            print("\nResult Summary: [NON-HOSTILE]")
        else:
            # Highlight only the hostile ones in the final summary
            actual_hits = [l for l in triggered_labels if l != "non-hostile"]
            print(f"\nResult Summary: HOSTILE ({', '.join(actual_hits)})")

if __name__ == "__main__":
    main()
