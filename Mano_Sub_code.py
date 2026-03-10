#!/usr/bin/env python3
"""
Telugu Prompt-Style Recovery — clean rewrite
Python 3.12 + PyTorch 2.10 + Transformers 4.40
Run: /opt/homebrew/Caskroom/miniforge/base/bin/python3 telugu_style_fixed.py
"""
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

# ── paths ────────────────────────────────────────────────────────
BASE_DIR        = "/Users/itscs/Prompt Recovery for LLM in Telugu"
TRAIN_PATH      = os.path.join(BASE_DIR, "train.csv")
DEV_PATH        = os.path.join(BASE_DIR, "dev.csv")
TEST_PATH       = os.path.join(BASE_DIR, "test_unlabeled.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")

# ── config ───────────────────────────────────────────────────────
MODEL_NAME   = "google/muril-base-cased"
MAX_LENGTH   = 512
BATCH_SIZE   = 8
EPOCHS       = 10
PATIENCE     = 4
N_FOLDS      = 5
WARMUP_STEPS = 200   # classifier-only warmup steps per fold

VALID_LABELS = [
    "Formal","Informal","Optimistic","Pessimistic",
    "Humorous","Serious","Inspiring","Authoritative","Persuasive",
]

# ── device (force CPU — MPS has gradient bugs) ───────────────────
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── tokeniser ────────────────────────────────────────────────────
print("Loading tokeniser...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── helpers ──────────────────────────────────────────────────────
def clean(v):
    return str(v).strip() if v is not None else ""

def read_csv(path):
    for enc in ("utf-8-sig","utf-8","cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, encoding_errors="replace")
        except Exception:
            continue
    raise IOError(f"Cannot read {path}")

def load(path, labeled=True):
    df = read_csv(path).reset_index(drop=True)
    df.columns = [clean(c) for c in df.columns]
    print(f"  cols: {df.columns.tolist()}")
    # rename columns
    rn = {}
    for c in df.columns:
        if c == "CHANGE STYLE":       rn[c] = "changed"
        elif c == "ORIGINAL TRANSCRIPTS": rn[c] = "original"
        elif labeled and c == "STYLE": rn[c] = "label"
    df.rename(columns=rn, inplace=True)
    if labeled:
        df["label"] = [clean(v) for v in df["label"].tolist()]
        df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    df["changed"]  = [clean(v) for v in df["changed"].tolist()]
    df["original"] = [clean(v) for v in df["original"].tolist()]
    df = df[df["changed"].apply(lambda x: len(x) > 3)].reset_index(drop=True)
    print(f"  {len(df)} rows loaded")
    return df

# ── tokenise: original + changed as sentence pair ────────────────
def tokenise_batch(originals, changeds):
    """Tokenise a batch of (original, changed) pairs."""
    return tokenizer(
        originals,
        changeds,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

def make_tensors(df):
    # Use changed text only (single text input)
    texts  = df["changed"].tolist()
    labels = torch.tensor(df["label_idx"].tolist(), dtype=torch.long)
    all_ids, all_mask = [], []
    for i in range(0, len(texts), 128):
        enc = tokenizer(
            [str(x) for x in texts[i:i+128]],
            padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        all_ids.append(enc["input_ids"])
        all_mask.append(enc["attention_mask"])
    return TensorDataset(torch.cat(all_ids), torch.cat(all_mask), labels)

def make_loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

# ── model ────────────────────────────────────────────────────────
def build_model(n):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n, ignore_mismatched_sizes=True).to(DEVICE)

# ── two-phase training ───────────────────────────────────────────
def phase1_warmup(model, loader, warmup_epochs=3):
    """No-op: warmup disabled, using discriminative LR instead."""
    pass

def get_optimizer(model):
    nd = ["bias", "LayerNorm.weight"]
    # Very high LR for classifier (random init), low for encoder (pretrained)
    classifier = [(n,p) for n,p in model.named_parameters() if "classifier" in n]
    top_layers  = [(n,p) for n,p in model.named_parameters()
                   if any(f"layer.{i}" in n for i in [9,10,11]) or "pooler" in n]
    rest        = [(n,p) for n,p in model.named_parameters()
                   if "classifier" not in n and not any(f"layer.{i}" in n for i in [9,10,11]) and "pooler" not in n]
    def pg(params, lr):
        return [
            {"params":[p for n,p in params if not any(x in n for x in nd)], "lr":lr, "weight_decay":0.01},
            {"params":[p for n,p in params if     any(x in n for x in nd)], "lr":lr, "weight_decay":0.0},
        ]
    return AdamW(pg(classifier, 1e-3) + pg(top_layers, 3e-5) + pg(rest, 1e-5))

# ── training ─────────────────────────────────────────────────────
def train_fold(tr_df, va_df, fold, le):
    print(f"\n{'='*55}\n  FOLD {fold}  train={len(tr_df)}  val={len(va_df)}\n{'='*55}")
    train_loader = make_loader(make_tensors(tr_df), shuffle=True)
    val_loader   = make_loader(make_tensors(va_df))
    model        = build_model(len(le.classes_))
    phase1_warmup(model, train_loader, warmup_epochs=3)
    optimizer    = get_optimizer(model)
    criterion    = nn.CrossEntropyLoss()
    best_f1, pat = 0.0, 0
    best_preds, best_acts = [], []
    save_path = os.path.join(BASE_DIR, f"fold{fold}_best.pt")

    for epoch in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"  Ep {epoch+1}/{EPOCHS}")
        for ids, mask, lbls in pbar:
            ids, mask, lbls = ids.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(input_ids=ids, attention_mask=mask).logits, lbls)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item(); n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for ids, mask, lbls in val_loader:
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                preds.extend(model(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu().tolist())
                acts.extend(lbls.tolist())

        f1 = f1_score(acts, preds, average="macro", zero_division=0)
        print(f"\n  Ep {epoch+1}  loss={tot/max(n,1):.4f}  macro_f1={f1:.4f}")
        print(classification_report(acts, preds, target_names=[str(c) for c in le.classes_], zero_division=0))

        if f1 > best_f1:
            best_f1, pat = f1, 0
            best_preds, best_acts = preds[:], acts[:]
            torch.save(model.state_dict(), save_path)
            print(f"  ** Best saved (F1={best_f1:.4f})")
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"  Early stop"); break

    del model; return best_f1, best_acts, best_preds

def run_kfold(df, le):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    f1s, oa, op = [], [], []
    for fold, (ti, vi) in enumerate(skf.split(df["changed"], df["label_idx"]), 1):
        f1, a, p = train_fold(df.iloc[ti].reset_index(drop=True),
                               df.iloc[vi].reset_index(drop=True), fold, le)
        f1s.append(f1); oa.extend(a); op.extend(p)
    print(f"\n{'='*55}\n  CV SUMMARY\n{'='*55}")
    for i,f in enumerate(f1s,1): print(f"  Fold {i}: {f:.4f}")
    print(f"  Mean={np.mean(f1s):.4f}  Std={np.std(f1s):.4f}")
    print(classification_report(oa, op, target_names=[str(c) for c in le.classes_], zero_division=0))
    pd.DataFrame({"fold":range(1,N_FOLDS+1),"f1":f1s}).to_csv(
        os.path.join(BASE_DIR,"kfold_log.csv"), index=False)
    return f1s

def train_final(df, le):
    print(f"\n{'='*55}\n  FINAL MODEL\n{'='*55}")
    loader = make_loader(make_tensors(df), shuffle=True)
    model  = build_model(len(le.classes_))
    phase1_warmup(model, loader, warmup_epochs=3)
    opt    = get_optimizer(model)
    crit   = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train(); tot, n = 0.0, 0
        for ids, mask, lbls in tqdm(loader, desc=f"  Final Ep {epoch+1}"):
            ids, mask, lbls = ids.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(input_ids=ids, attention_mask=mask).logits, lbls)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); n += 1
        print(f"  Ep {epoch+1} loss={tot/max(n,1):.4f}")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(FINAL_MODEL_DIR,"model.pt"))
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"  Saved -> {FINAL_MODEL_DIR}")
    return model

def make_submission(model, le):
    print("\nMaking submission...")
    try:
        df = read_csv(TEST_PATH).reset_index(drop=True)
        df.columns = [clean(c) for c in df.columns]
        orig_col = next((c for c in df.columns if "ORIGINAL" in c.upper()), df.columns[1])
        chgd_col = next((c for c in df.columns if "CHANGE" in c.upper()), df.columns[2])
        origs = [clean(v) for v in df[orig_col].tolist()]
        chgds = [clean(v) for v in df[chgd_col].tolist()]
        label_map = {i:str(l) for i,l in enumerate(le.classes_)}
        preds = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0,len(origs),BATCH_SIZE), desc="Inference"):
                enc = tokenise_batch(origs[i:i+BATCH_SIZE], chgds[i:i+BATCH_SIZE])
                enc = {k:v.to(DEVICE) for k,v in enc.items()}
                preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
        id_col = "ID" if "ID" in df.columns else df.columns[0]
        df["STYLE"] = [label_map[p] for p in preds]
        df[[id_col,"STYLE"]].to_csv(SUBMISSION_PATH, index=False)
        print(f"Saved -> {SUBMISSION_PATH}")
        dist = df["STYLE"].value_counts()
        print(dist.to_string())
    except Exception as e:
        import traceback; traceback.print_exc()

def main():
    print("="*55)
    print(f"  Telugu Style | {MODEL_NAME}")
    print(f"  Folds={N_FOLDS} Epochs={EPOCHS} Batch={BATCH_SIZE} MaxLen={MAX_LENGTH}")
    print("="*55)

    train_df = load(TRAIN_PATH)
    full_df  = pd.concat([train_df, load(DEV_PATH)], ignore_index=True) if os.path.exists(DEV_PATH) else train_df
    print(f"Total: {len(full_df)} rows")
    print(full_df["label"].value_counts().to_string())

    le = LabelEncoder()
    le.fit(sorted(VALID_LABELS))
    full_df["label_idx"] = le.transform(full_df["label"]).astype(int)
    label_map = {i:str(l) for i,l in enumerate(le.classes_)}
    print(f"Classes: {list(le.classes_)}")

    print("\n[SANITY] Forward pass check...")
    tmp = build_model(9)
    row = full_df.iloc[:2]
    enc = tokenizer([str(x) for x in row["changed"].tolist()],
                    padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        out = tmp(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    print(f"  logits shape: {out.logits.shape}  std: {out.logits.std().item():.4f} — OK")
    del tmp

    f1s = run_kfold(full_df, le)
    print(f"\nCV Mean F1={np.mean(f1s):.4f}")
    make_submission(train_final(full_df, le), le)

if __name__ == "__main__":
    main()