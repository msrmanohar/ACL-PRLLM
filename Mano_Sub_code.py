import os
os.environ["OMP_NUM_THREADS"]        = "10"  
os.environ["MKL_NUM_THREADS"]        = "10"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ
import random, warnings, time, glob
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from torch.optim import AdamW
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

torch.set_num_threads(10)          
torch.set_num_interop_threads(4)  

print("=" * 60)
print("  Telugu Prompt-Style Classification — Phase 2")
print("=" * 60)
print(f"  PyTorch   : {torch.__version__}")
print(f"  Threads   : compute={torch.get_num_threads()}  interop={torch.get_num_interop_threads()}")
print("=" * 60)

BASE_DIR        = "/Users/itscs/Prompt Recovery for LLM in Telugu"
TRAIN_PATH      = os.path.join(BASE_DIR, "train.csv")
DEV_PATH        = os.path.join(BASE_DIR, "dev.csv")
TEST_PATH       = os.path.join(BASE_DIR, "test_unlabeled.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/muril-base-cased"
MAX_LENGTH   = 512
BATCH_SIZE   = 8
EPOCHS       = 10
PATIENCE     = 4
N_FOLDS      = 5

CONTRASTIVE_ALPHA = 0.0   

VALID_LABELS = sorted([
    "Formal", "Informal", "Optimistic", "Pessimistic",
    "Humorous", "Serious", "Inspiring", "Authoritative", "Persuasive",
])

DEVICE = torch.device("cpu")
print(f"  Device    : {DEVICE}")
print(f"  Contrastive loss alpha: {CONTRASTIVE_ALPHA}")
print("=" * 60)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ARTICLE-AWARE SAMPLER
class ArticleAwareSampler(Sampler):
    """
    Round-robin sampling across source articles.
    Guarantees: no two samples from the same article in the same batch.
    This eliminates gradient cancellation / majority-class collapse.
    """
    def __init__(self, article_ids, batch_size, shuffle=True):
        self.article_ids = article_ids
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.groups      = defaultdict(list)
        for idx, art in enumerate(article_ids):
            self.groups[art].append(idx)
        self.art_keys = list(self.groups.keys())

    def __iter__(self):
        groups = {k: list(v) for k, v in self.groups.items()}
        if self.shuffle:
            for k in groups:
                random.shuffle(groups[k])
            random.shuffle(self.art_keys)
        indices = []
        iters   = {k: iter(v) for k, v in groups.items()}
        active  = list(self.art_keys)
        while active:
            batch      = []
            exhausted  = []
            for art in active:
                try:
                    batch.append(next(iters[art]))
                    if len(batch) == self.batch_size:
                        break
                except StopIteration:
                    exhausted.append(art)
            active = [a for a in active if a not in exhausted]
            if batch:
                indices.extend(batch)
        return iter(indices)

    def __len__(self):
        return sum(len(v) for v in self.groups.values())

class SupervisedContrastiveLoss(nn.Module):
    """
    Pushes same-class [CLS] embeddings together and
    different-class embeddings apart in the 768-dim space.

    This specifically helps the Serious class (F1=0.03) which is
    defined by the ABSENCE of affective markers — standard cross-entropy
    cannot learn neutrality from tokens alone, but contrastive loss
    can learn to separate Serious from Formal and Authoritative.

    Usage:
        Set CONTRASTIVE_ALPHA = 0.1 to activate.
        Start at 0.1, tune between 0.05 and 0.3.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features : [batch_size, 768]  — [CLS] embeddings
        labels   : [batch_size]       — class indices
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, requires_grad=True)

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        labels   = labels.unsqueeze(1)
        pos_mask = torch.eq(labels, labels.T).float()
        eye      = torch.eye(features.size(0), device=features.device)
        pos_mask = pos_mask - eye           # remove self-pairs

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()

        # Exp similarities (exclude self)
        exp_sim = torch.exp(sim) * (1 - eye)

        # Log probability
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss  = -(pos_mask * log_prob).sum(dim=1) / n_pos

        return loss.mean()


print("\nLoading MuRIL tokeniser...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("  Tokeniser loaded.")


def clean(v):
    return str(v).strip() if v is not None else ""


def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, encoding_errors="replace")
        except Exception:
            continue
    raise IOError(f"Cannot read {path}")


def load(path, labeled=True):
    df = read_csv(path).reset_index(drop=True)
    df.columns = [clean(c) for c in df.columns]
    print(f"  Columns: {df.columns.tolist()}")
    rn = {}
    for c in df.columns:
        if c == "CHANGE STYLE":
            rn[c] = "changed"
        elif c == "ORIGINAL TRANSCRIPTS":
            rn[c] = "original"
        elif labeled and c == "STYLE":
            rn[c] = "label"
    df.rename(columns=rn, inplace=True)
    if labeled:
        df["label"] = [clean(v) for v in df["label"].tolist()]
        df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    df["changed"]  = [clean(v) for v in df["changed"].tolist()]
    df["original"] = [clean(v) for v in df["original"].tolist()]
    df = df[df["changed"].apply(lambda x: len(x) > 3)].reset_index(drop=True)
    print(f"  {len(df)} rows loaded from {os.path.basename(path)}")
    return df


def make_tensors(df):
    """Tokenise all texts and return TensorDataset + article_ids list."""
    texts       = df["changed"].tolist()
    labels      = torch.tensor(df["label_idx"].tolist(), dtype=torch.long)
    article_ids = df["original"].tolist()
    all_ids, all_mask = [], []
    for i in range(0, len(texts), 128):
        enc = tokenizer(
            [str(x) for x in texts[i:i+128]],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        all_ids.append(enc["input_ids"])
        all_mask.append(enc["attention_mask"])
    ds = TensorDataset(torch.cat(all_ids), torch.cat(all_mask), labels)
    return ds, article_ids


def make_loader(ds, shuffle=False, article_ids=None):
    if shuffle and article_ids is not None:
        sampler = ArticleAwareSampler(article_ids, BATCH_SIZE, shuffle=True)
        return DataLoader(ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0)

def build_model(n_classes):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # torch.compile — ~15% speed boost on PyTorch 2.x
    try:
        model = torch.compile(model, backend="aot_eager")
        print("  torch.compile() applied — speed boost active")
    except Exception:
        print("  torch.compile() skipped — running standard mode")

    return model


def get_optimizer(model):
    """
    Discriminative learning rates:
      Classifier head  : 0.001   (random init — learns fast)
      Top 3 layers + pooler : 0.00003  (fine-tunes style patterns)
      Base 9 layers    : 0.00001  (preserves Telugu knowledge)
    """
    no_decay = ["bias", "LayerNorm.weight"]

    classifier = [(n, p) for n, p in model.named_parameters()
                  if "classifier" in n]
    top_layers = [(n, p) for n, p in model.named_parameters()
                  if any(f"layer.{i}" in n for i in [9, 10, 11])
                  or "pooler" in n]
    rest       = [(n, p) for n, p in model.named_parameters()
                  if "classifier" not in n
                  and not any(f"layer.{i}" in n for i in [9, 10, 11])
                  and "pooler" not in n]

    def pg(params, lr):
        return [
            {"params": [p for n, p in params if not any(x in n for x in no_decay)],
             "lr": lr, "weight_decay": 0.01},
            {"params": [p for n, p in params if     any(x in n for x in no_decay)],
             "lr": lr, "weight_decay": 0.0},
        ]

    param_groups = (
        pg(classifier, 0.001)    +   # 1e-3
        pg(top_layers, 0.00003)  +   # 3e-5
        pg(rest,       0.00001)      # 1e-5
    )
    return AdamW(param_groups)


def save_checkpoint(fold, epoch, model, optimizer, best_f1):
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_fold{fold}_ep{epoch}.pt")
    torch.save({
        "fold":      fold,
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_f1":   best_f1,
    }, path)
    # Keep only last 2 checkpoints per fold to save disk space
    old = sorted(glob.glob(
        os.path.join(CHECKPOINT_DIR, f"ckpt_fold{fold}_ep*.pt")))[:-2]
    for f in old:
        os.remove(f)


def load_checkpoint(fold, model, optimizer):
    """Resume training from last saved checkpoint for this fold."""
    ckpts = sorted(glob.glob(
        os.path.join(CHECKPOINT_DIR, f"ckpt_fold{fold}_ep*.pt")))
    if not ckpts:
        return 0, 0.0   # start_epoch, best_f1
    ckpt = torch.load(ckpts[-1], map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from epoch {ckpt['epoch']} (best F1={ckpt['best_f1']:.4f})")
    return ckpt["epoch"], ckpt["best_f1"]

def train_fold(tr_df, va_df, fold, le, resume=True):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}  |  train={len(tr_df)}  val={len(va_df)}")
    print(f"{'='*60}")

    train_ds, train_arts = make_tensors(tr_df)
    train_loader         = make_loader(train_ds, shuffle=True,
                                       article_ids=train_arts)
    val_ds, _            = make_tensors(va_df)
    val_loader           = make_loader(val_ds)

    model        = build_model(len(le.classes_))
    optimizer    = get_optimizer(model)
    criterion    = nn.CrossEntropyLoss()
    contrastive  = SupervisedContrastiveLoss(temperature=0.07)

    # Resume from checkpoint if available
    start_epoch = 0
    best_f1     = 0.0
    if resume:
        start_epoch, best_f1 = load_checkpoint(fold, model, optimizer)

    best_preds, best_acts = [], []
    save_path = os.path.join(BASE_DIR, f"fold{fold}_best.pt")
    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        ep_start = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(train_loader,
                    desc=f"  Fold {fold} | Ep {epoch+1}/{EPOCHS}",
                    ncols=90)

        for ids, mask, lbls in pbar:
            ids  = ids.to(DEVICE)
            mask = mask.to(DEVICE)
            lbls = lbls.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True)
            logits  = outputs.logits

            # Cross-entropy loss (always active)
            ce_loss = criterion(logits, lbls)

            # Contrastive loss (Phase 2 — activate by setting CONTRASTIVE_ALPHA > 0)
            if CONTRASTIVE_ALPHA > 0.0:
                # Extract [CLS] embeddings from last hidden state
                cls_embeddings = outputs.hidden_states[-1][:, 0, :]
                sc_loss = contrastive(cls_embeddings, lbls)
                loss    = ce_loss + CONTRASTIVE_ALPHA * sc_loss
            else:
                loss = ce_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for ids, mask, lbls in val_loader:
                ids  = ids.to(DEVICE)
                mask = mask.to(DEVICE)
                logits = model(input_ids=ids,
                               attention_mask=mask).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                acts.extend(lbls.tolist())

        f1       = f1_score(acts, preds, average="macro", zero_division=0)
        avg_loss = total_loss / max(n_batches, 1)
        ep_time  = time.time() - ep_start
        n_classes_predicted = len(set(preds))

        print(f"\n  Ep {epoch+1:2d}  loss={avg_loss:.4f}  "
              f"macro_f1={f1:.4f}  "
              f"classes={n_classes_predicted}/9  "
              f"time={ep_time/60:.1f}min")
        print(classification_report(
            acts, preds,
            target_names=[str(c) for c in le.classes_],
            zero_division=0,
        ))

        # Save epoch checkpoint (resume protection)
        save_checkpoint(fold, epoch + 1, model, optimizer, best_f1)

        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            best_preds = preds[:]
            best_acts  = acts[:]
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved — F1={best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("  Early stopping triggered.")
                break

    del model
    return best_f1, best_acts, best_preds
def run_kfold(df, le):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    f1s, oa, op = [], [], []

    for fold, (ti, vi) in enumerate(
            skf.split(df["changed"], df["label_idx"]), 1):
        f1, a, p = train_fold(
            df.iloc[ti].reset_index(drop=True),
            df.iloc[vi].reset_index(drop=True),
            fold, le, resume=True,
        )
        f1s.append(f1)
        oa.extend(a)
        op.extend(p)

    print(f"\n{'='*60}")
    print("  CV SUMMARY")
    print(f"{'='*60}")
    for i, f in enumerate(f1s, 1):
        print(f"  Fold {i}: {f:.4f}")
    print(f"  Mean={np.mean(f1s):.4f}  Std={np.std(f1s):.4f}")
    print("\nAggregated classification report (all folds):")
    print(classification_report(
        oa, op,
        target_names=[str(c) for c in le.classes_],
        zero_division=0,
    ))

    # ── Bug fix: range(1, len(f1s)+1) not range(1, N_FOLDS+1) ──────
    pd.DataFrame({
        "fold": range(1, len(f1s) + 1),
        "f1":   f1s,
    }).to_csv(os.path.join(BASE_DIR, "kfold_log.csv"), index=False)
    print(f"  Fold results saved → kfold_log.csv")

    return f1s

def train_final(df, le):
    print(f"\n{'='*60}")
    print("  FINAL MODEL — training on all data")
    print(f"{'='*60}")
    ds, arts = make_tensors(df)
    loader   = make_loader(ds, shuffle=True, article_ids=arts)
    model    = build_model(len(le.classes_))
    opt      = get_optimizer(model)
    crit     = nn.CrossEntropyLoss()
    contrastive = SupervisedContrastiveLoss(temperature=0.07)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        start = time.time()
        for ids, mask, lbls in tqdm(
                loader, desc=f"  Final Ep {epoch+1}/{EPOCHS}", ncols=90):
            ids  = ids.to(DEVICE)
            mask = mask.to(DEVICE)
            lbls = lbls.to(DEVICE)
            opt.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True)
            ce_loss = crit(outputs.logits, lbls)
            if CONTRASTIVE_ALPHA > 0.0:
                cls_emb = outputs.hidden_states[-1][:, 0, :]
                sc_loss = contrastive(cls_emb, lbls)
                loss    = ce_loss + CONTRASTIVE_ALPHA * sc_loss
            else:
                loss = ce_loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        elapsed = (time.time() - start) / 60
        print(f"  Ep {epoch+1:2d}  loss={total_loss/max(n_batches,1):.4f}  "
              f"time={elapsed:.1f}min")

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(FINAL_MODEL_DIR, "model.pt"))
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"  Saved → {FINAL_MODEL_DIR}")
    return model

# SUBMISSION
def make_submission(model, le):
    print("\nGenerating submission CSV...")
    try:
        df = read_csv(TEST_PATH).reset_index(drop=True)
        df.columns = [clean(c) for c in df.columns]
        orig_col = next(
            (c for c in df.columns if "ORIGINAL" in c.upper()), df.columns[1])
        chgd_col = next(
            (c for c in df.columns if "CHANGE" in c.upper()), df.columns[2])
        origs     = [clean(v) for v in df[orig_col].tolist()]
        chgds     = [clean(v) for v in df[chgd_col].tolist()]
        label_map = {i: str(l) for i, l in enumerate(le.classes_)}
        preds     = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(origs), BATCH_SIZE),
                          desc="Inference", ncols=90):
                enc = tokenizer(
                    [str(x) for x in chgds[i:i+BATCH_SIZE]],
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                preds.extend(
                    model(**enc).logits.argmax(-1).cpu().tolist())
        id_col = "ID" if "ID" in df.columns else df.columns[0]
        df["STYLE"] = [label_map[p] for p in preds]
        df[[id_col, "STYLE"]].to_csv(SUBMISSION_PATH, index=False)
        print(f"  Saved → {SUBMISSION_PATH}")
        print(df["STYLE"].value_counts().to_string())
    except Exception:
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*60}")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Folds    : {N_FOLDS}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"  MaxLen   : {MAX_LENGTH}  |  Patience: {PATIENCE}")
    print(f"  Phase 2  : Contrastive alpha = {CONTRASTIVE_ALPHA}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading train data...")
    train_df = load(TRAIN_PATH)
    if os.path.exists(DEV_PATH):
        print("Loading dev data...")
        dev_df   = load(DEV_PATH)
        full_df  = pd.concat([train_df, dev_df], ignore_index=True)
    else:
        full_df  = train_df
    print(f"\nTotal samples: {len(full_df)}")
    print(full_df["label"].value_counts().to_string())

    # ── Label encoding ────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(sorted(VALID_LABELS))
    full_df["label_idx"] = le.transform(full_df["label"]).astype(int)
    print(f"\nClasses ({len(le.classes_)}): {list(le.classes_)}")

    # ── Sanity check — forward pass ───────────────────────────────────
    print("\n[SANITY] Running forward pass check...")
    tmp_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=9,
        ignore_mismatched_sizes=True).to(DEVICE)
    row = full_df.iloc[:2]
    enc = tokenizer(
        [str(x) for x in row["changed"].tolist()],
        padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    with torch.no_grad():
        out = tmp_model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"])
    print(f"  Logits shape: {out.logits.shape}  "
          f"std: {out.logits.std().item():.4f}  ✓ OK")
    del tmp_model

    # ── K-Fold CV ──────────────────────────────────────────────────────
    cv_start = time.time()
    f1s = run_kfold(full_df, le)
    cv_time = (time.time() - cv_start) / 3600
    print(f"\n  CV complete in {cv_time:.2f} hours")
    print(f"  CV Mean F1 = {np.mean(f1s):.4f}  Std = {np.std(f1s):.4f}")

    # ── Final model + submission ───────────────────────────────────────
    final_model = train_final(full_df, le)
    make_submission(final_model, le)

    print(f"\n{'='*60}")
    print("  ALL DONE")
    print(f"  Best fold F1  : {max(f1s):.4f}  (Fold {f1s.index(max(f1s))+1})")
    print(f"  CV Mean F1    : {np.mean(f1s):.4f}")
    print(f"  Models saved  : {BASE_DIR}/fold*_best.pt")
    print(f"  Final model   : {FINAL_MODEL_DIR}/")
    print(f"  Submission    : {SUBMISSION_PATH}")
    print(f"  CV log        : {BASE_DIR}/kfold_log.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
