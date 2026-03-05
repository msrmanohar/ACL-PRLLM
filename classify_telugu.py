import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
 
# 1. MAC & STABILITY SETUP
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_everything()
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 2. CONFIGURATION
MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 256  
BATCH_SIZE = 16   
EPOCHS = 10       
# Higher learning rate is needed because we are only training the "Head"
#HEAD_LR = 1e-3
#HEAD_LR = 1e-4 
HEAD_LR = 2e-5 
# 3. BALANCED DATA LOADING
def load_and_balance(filename, is_train=True):
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
    except:
        df = pd.read_csv(filename, encoding='cp1252', encoding_errors='replace')
    
    df.columns = df.columns.str.strip()
    df.rename(columns={'ORIGINAL TRANSCRIPTS': 'text', 'STYLE': 'label'}, inplace=True, errors='ignore')
    df = df.dropna(subset=['text', 'label'])
    
    if is_train:
        print("⚖️ Balancing classes via oversampling...")
        major_size = df['label'].value_counts().max()
        balanced_list = []
        for cls in df['label'].unique():
            class_subset = df[df['label'] == cls]
            resampled_subset = resample(class_subset, replace=True, n_samples=major_size, random_state=42)
            balanced_list.append(resampled_subset)
        df = pd.concat(balanced_list).sample(frac=1).reset_index(drop=True)
    return df

train_df = load_and_balance("train.csv", is_train=True)
dev_df = load_and_balance("dev.csv", is_train=False)

le = LabelEncoder()
train_df['label_idx'] = le.fit_transform(train_df['label'])
dev_df['label_idx'] = le.transform(dev_df['label'])
label_map = {i: label for i, label in enumerate(le.classes_)}

# 4. TOKENIZATION
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_ds = Dataset.from_pandas(train_df[['text', 'label_idx']]).map(tokenize_fn, batched=True)
dev_ds = Dataset.from_pandas(dev_df[['text', 'label_idx']]).map(tokenize_fn, batched=True)
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label_idx"])
dev_ds.set_format("torch", columns=["input_ids", "attention_mask", "label_idx"])

train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)

# 5. MODEL & FROZEN TRAINING
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)

# --- FREEZE BASE LAYERS ---
# We keep the "Brain" of MuRIL fixed and only train the "Mouth" (Classifier)
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# Calculate weights to punish majority-class spamming
classes = np.unique(train_df['label_idx'])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label_idx'].values)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR)

best_f1 = 0.0
print(f"❄️ FROZEN HEAD TRAINING: Only training the classifier on {device}...")

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
    
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = criterion(outputs.logits, batch['label_idx'])
        
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Evaluation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            actuals.extend(batch['label_idx'].cpu().numpy())
    
    val_f1 = f1_score(actuals, preds, average='macro')
    print(f"\n📊 Epoch {epoch+1} Results (Macro F1: {val_f1:.4f}):")
    print(classification_report(actuals, preds, target_names=le.classes_, zero_division=0))

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        print("⭐ Saved New Best Model!")

# 6. FINAL SUBMISSION
# 6. FINAL SUBMISSION (STABILIZED)
print("\n🔮 Creating submission.csv...")
try:
    test_df = pd.read_csv("test_unlabeled.csv")
    
    # Ensure all inputs are strings and handle potential NaNs
    # Most common cause of your error is a blank row in the CSV
    test_texts = test_df.iloc[:, 1].fillna("").astype(str).tolist()
    
    if os.path.exists("best_model.pt"):
        print("✅ Loading best weights from Epoch 9...")
        model.load_state_dict(torch.load("best_model.pt"))
    
    model.eval()
    final_preds = []
    
    # Process in batches to avoid memory issues on Mac
    for i in tqdm(range(0, len(test_texts), BATCH_SIZE), desc="Final Inference"):
        batch_text = test_texts[i : i + BATCH_SIZE]
        
        # Double check batch is not empty
        if not batch_text: continue
            
        inputs = tokenizer(
            batch_text, 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs).logits
            # Convert numeric IDs back to style names
            batch_preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            final_preds.extend([label_map[p] for p in batch_preds])
            
    # Match predictions to the original ID column
    test_df['STYLE'] = final_preds
    test_df[['ID', 'STYLE']].to_csv("submission.csv", index=False)
    print("🏁 Done! 'submission.csv' is saved and ready for upload.")

except Exception as e:
    print(f"❌ Final prediction failed: {e}")
    import traceback
    traceback.print_exc()