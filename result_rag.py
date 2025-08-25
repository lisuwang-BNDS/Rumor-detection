import json
from tqdm import tqdm

# Read data from rumor\data\generated_predictions.jsonl
with open(r'data\web_search_rag_predict.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Convert each line to JSON format
data = [json.loads(line) for line in lines]

# Convert "yes" and "no" in label and predict to 1 and 0
labels = [d['label']  for d in data]
preds = [1 if "yes" in str(d['predict']).lower() else 0 for d in data]

# Alternative conversion method (commented out)
# labels = [int(d['label'])  for d in data]
# preds = [int(d['predict']) for d in data]

# Calculate accuracy, precision, recall, and F1 score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
# Calculate F1 score for negative predictions
neg_preds = [1 if p == 0 else 0 for p in preds]
neg_labels = [1 if l == 0 else 0 for l in labels]
neg_f1 = f1_score(neg_labels, neg_preds)
# Calculate F1 score for positive predictions
f1 = f1_score(labels, preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Negative F1: {neg_f1:.4f}')
print(f'F1: {f1:.4f}')
