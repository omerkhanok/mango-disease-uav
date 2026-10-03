
import warnings
warnings.filterwarnings("ignore")
import os, hashlib, copy, random, shutil, time
os.environ["PYTHONWARNINGS"] = "ignore"
from PIL import Image, ImageEnhance, ImageFilter
Image.MAX_IMAGE_PIXELS = None

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F          # ← needed for F.normalize in embeddings
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import (
    MulticlassF1Score, MulticlassConfusionMatrix,
    MulticlassPrecision, MulticlassRecall)
import timm

print("✓ Imports OK")
print(f"✓ PyTorch {torch.__version__}  timm {timm.__version__}")


# ─────────────────────────────────────────────────────────────
# SEED & DEVICE
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM   : "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────
# PATHS  ← only SYNTHETIC_DATA needs changing
# ─────────────────────────────────────────────────────────────
REAL_DATA      = Path("/kaggle/input/mango-dataset")
SYNTHETIC_DATA = Path("/kaggle/input/datasets/omerkhanok/final-synthetic/Synthetic Dataset/working/synthetic_drone")  # ← CHANGE

AUGMENTED_DIR  = Path("/kaggle/working/augmented")
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
LOG_DIR        = Path("/kaggle/working/logs")
for d in [AUGMENTED_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
MODEL_NAME   = "deit_small_patch16_224"
CLF_IMG_SIZE = 224
EMBED_DIM    = 384

if torch.cuda.is_available():
    _vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
    CLF_BATCH = 32 if _vram >= 15 else 16 if _vram >= 10 else 8
else:
    CLF_BATCH = 8

# ── Dataset split ─────────────────────────────────────────────
TRAIN_RATIO = 0.65
VAL_RATIO   = 0.20
TEST_RATIO  = 0.15

SYN_TRAIN_PER_CLASS = 100   # 800 total
SYN_VAL_PER_CLASS   = 100   # 800 total
SYN_TEST_PER_CLASS  = 25    # 200 total

TRAIN_AUG_COPIES = 3   # same for both → equal difficulty
VAL_AUG_COPIES   = 3

# ── 2-Phase training (NO Phase 3, NO Mixup) ───────────────────
CLF_EPOCHS    = 60
CLF_WARMUP_EP = 5     # short warmup — no mixup so learning is clean

PHASE1_END    = 15    # head only:     ep  1-30
PHASE2_START  = 16    # top-2 blocks:  ep 31-60

# ── Learning rates ────────────────────────────────────────────
CLF_LR_HEAD     = 5e-4    # head LR
CLF_LR_BACKBONE = 1e-6    # micro backbone LR (gap fix)

# ── Regularization ───────────────────────────────────────────
WD_PHASE1        = 0.05
WD_PHASE2        = 0.10
CLF_LABEL_SMOOTH = 0.1
CLF_DROP_RATE    = 0.1
CLF_DROP_PATH    = 0.1

# ── Head architecture ─────────────────────────────────────────
HEAD_DIM1  = 256
HEAD_DIM2  = 128
HEAD_DROP1 = 0.25
HEAD_DROP2 = 0.15

# ── Other ─────────────────────────────────────────────────────
CLF_SAVE_EVERY      = 5
EARLY_STOP_PATIENCE = 15
PHASE2_BLOCKS       = 2

# ── NO MIXUP ─────────────────────────────────────────────────
# Mixup was proven to be the sole cause of the 10-13% gap.
USE_MIXUP = False

# ── Leakage detection ─────────────────────────────────────────
# Cosine-similarity threshold for embedding-based near-duplicate
# detection. 1.0 = identical embedding, 0.95 catches images
# that differ only by slight augmentation / JPEG re-compression.
SIM_THRESHOLD = 0.95

print(f"\n{'='*58}")
print(f"  FINAL CONFIGURATION")
print(f"{'='*58}")
print(f"  Model         : {MODEL_NAME} + LayerNorm Head")
print(f"  Split         : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
print(f"  Batch         : {CLF_BATCH}")
print(f"  Aug copies    : train×{TRAIN_AUG_COPIES}  val×{VAL_AUG_COPIES}")
print(f"  Phase 1       : ep  1-{PHASE1_END}  head only")
print(f"  Phase 2       : ep {PHASE2_START}-{CLF_EPOCHS}  "
      f"top-{PHASE2_BLOCKS} blocks, LR={CLF_LR_BACKBONE}")
print(f"  Mixup         : DISABLED (was causing 10-13% gap)")
print(f"  LR head       : {CLF_LR_HEAD}")
print(f"  LR backbone   : {CLF_LR_BACKBONE} (micro)")
print(f"  WD P1/P2      : {WD_PHASE1}/{WD_PHASE2}")
print(f"  Label smooth  : {CLF_LABEL_SMOOTH}")
print(f"  Leak detect   : hash + ResNet-18 cosine sim ≥ {SIM_THRESHOLD}")
print(f"{'='*58}")


# ─────────────────────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = sorted([d.name for d in REAL_DATA.iterdir() if d.is_dir()])
NUM_CLASSES  = len(CLASS_NAMES)
print(f"\n[INFO] Classes ({NUM_CLASSES}): {CLASS_NAMES}")


# ─────────────────────────────────────────────────────────────
# GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────
_DRIVE = None; _FOLDER_ID = None; _DRIVE_READY = False

def setup_gdrive():
    global _DRIVE, _FOLDER_ID, _DRIVE_READY
    print("\n[GDRIVE] Connecting...")
    try:
        from kaggle_secrets import UserSecretsClient
        _FOLDER_ID = UserSecretsClient().get_secret("GDRIVE_FOLDER_ID")
    except Exception:
        print("[GDRIVE] No secret — skipping."); return
    try:
        os.system("pip install -q pydrive2 2>/dev/null")
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        gauth = GoogleAuth(); gauth.CommandLineAuth()
        _DRIVE = GoogleDrive(gauth); _DRIVE_READY = True
        print("[GDRIVE] ✓ Connected.")
    except Exception as e:
        print(f"[GDRIVE] Failed: {e}")

def gdrive_upload(local_path, subfolder=None):
    if not _DRIVE_READY: return
    try:
        lp = Path(local_path)
        if not lp.exists(): return
        pid = _FOLDER_ID
        if subfolder:
            q  = (f"'{_FOLDER_ID}' in parents and title='{subfolder}' "
                  f"and mimeType='application/vnd.google-apps.folder' "
                  f"and trashed=false")
            fl = _DRIVE.ListFile({"q": q}).GetList()
            pid = fl[0]["id"] if fl else None
            if not pid:
                f = _DRIVE.CreateFile({
                    "title": subfolder,
                    "parents": [{"id": _FOLDER_ID}],
                    "mimeType": "application/vnd.google-apps.folder"})
                f.Upload(); pid = f["id"]
        q2  = f"'{pid}' in parents and title='{lp.name}' and trashed=false"
        fl2 = _DRIVE.ListFile({"q": q2}).GetList()
        gf  = fl2[0] if fl2 else _DRIVE.CreateFile(
            {"title": lp.name, "parents": [{"id": pid}]})
        gf.SetContentFile(str(lp)); gf.Upload()
        print(f"  [GDRIVE] ✓ {lp.name} ({lp.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  [GDRIVE] skip {Path(local_path).name}: {e}")


# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "training_log.txt"

def log(msg, also_print=True):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print: print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────
# IMAGE COLLECTION
# ─────────────────────────────────────────────────────────────
def _collect_images(folder):
    imgs = []
    for ext in ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG"):
        imgs.extend(Path(folder).glob(ext))
    return imgs


# ─────────────────────────────────────────────────────────────
# LEAKAGE FIX  (Hash  +  Embedding-based near-duplicate check)
# ─────────────────────────────────────────────────────────────

def _get_hash(path, chunk=8192):
    """MD5 hash for exact-duplicate detection."""
    h = hashlib.md5()
    try:
        with open(path,"rb") as f:
            while data := f.read(chunk):
                h.update(data)
        return h.hexdigest()
    except Exception:
        return None


# ── Embedding helpers ─────────────────────────────────────────

def _get_embedding_model():
    """
    Lightweight ResNet-18 used ONLY for leakage embedding.
    Returns L2-normalised 512-dim vectors.
    Loaded once, then discarded to free GPU memory.
    """
    model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    model.eval()
    return model.to(DEVICE)


_EMB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _extract_embeddings(paths, emb_model, batch_size=64, desc=""):
    """
    Extract L2-normalised embeddings for a list of image paths.
    Returns a CPU tensor of shape (N, D).
    """
    all_embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, leave=False):
        batch = paths[i : i + batch_size]
        imgs  = []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(_EMB_TRANSFORM(img))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            emb = emb_model(tensor)
        emb = F.normalize(emb, p=2, dim=1)    # unit vectors → dot = cosine
        all_embs.append(emb.cpu())
    if not all_embs:
        return torch.zeros(0, 512)
    return torch.cat(all_embs, dim=0)


def _visualize_similar_pairs(similar_pairs, top_n=12):
    """
    Save a grid of the most similar train / val-test pairs so
    you can eyeball whether the threshold is correct.
    """
    if not similar_pairs:
        return
    n   = min(top_n, len(similar_pairs))
    fig, axes = plt.subplots(n, 2, figsize=(7, n * 3))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"Top-{n} Near-Duplicate Pairs Removed from Train\n"
        f"Left = Train  |  Right = Val / Test  |  "
        f"Cosine Sim ≥ {SIM_THRESHOLD}",
        fontsize=10, fontweight="bold")

    for i, pair in enumerate(similar_pairs[:n]):
        for j, (key, role) in enumerate([("train", "Train"),
                                          ("matches", "Val/Test")]):
            ax = axes[i][j]
            try:
                img = Image.open(pair[key]).convert("RGB")
                img = img.resize((224, 224))
                ax.imshow(np.array(img))
            except Exception:
                ax.text(0.5, 0.5, "Load error",
                        ha="center", va="center", transform=ax.transAxes)
            ax.set_title(
                f"[{role}]  sim={pair['similarity']:.4f}\n"
                f"{Path(pair[key]).name[:30]}",
                fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    path = LOG_DIR / "similar_pairs_visualization.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    gdrive_upload(path, subfolder="logs")
    log(f"  [EMBED] Pair visualization saved → {path.name}")


# ── Main leakage function ─────────────────────────────────────

def fix_leakage(train_s, val_s, test_s):
    """
    Two-stage leakage removal:
      Stage 1 — MD5 hash  : catches exact binary duplicates.
      Stage 2 — ResNet-18 : catches near-duplicates (slight
                             crop / augmentation / re-save).

    Returns a clean train list with leaky images removed.
    """
    log("\n[LEAKAGE] ════════════════════════════════════════")
    log("[LEAKAGE] Stage 1 — MD5 hash (exact duplicates)")
    log("[LEAKAGE] ════════════════════════════════════════")

    # --- Stage 1: exact hash duplicates ---
    protected_hashes = {}
    for path, label in tqdm(val_s + test_s,
                            desc="  Hashing val+test", leave=False):
        h = _get_hash(path)
        if h:
            protected_hashes[h] = path

    hash_clean   = []
    hash_removed = []
    for path, label in tqdm(train_s,
                            desc="  Scanning train (hash)", leave=False):
        h = _get_hash(path)
        if h in protected_hashes:
            hash_removed.append({"train":   path,
                                  "matches": protected_hashes[h]})
        else:
            hash_clean.append((path, label))

    if hash_removed:
        log(f"  [HASH] Removed {len(hash_removed)} exact duplicates:")
        for r in hash_removed:
            log(f"    {Path(r['train']).name:<40} "
                f"↔ {Path(r['matches']).name}")
    else:
        log("  [HASH] ✓ No exact duplicates found")

    # --- Stage 2: embedding-based near-duplicate check ---
    log(f"\n[LEAKAGE] Stage 2 — ResNet-18 cosine similarity "
        f"(threshold ≥ {SIM_THRESHOLD})")
    log("[LEAKAGE] ════════════════════════════════════════")

    protected_paths = [p for p, _ in (val_s + test_s)]
    train_paths     = [p for p, _ in hash_clean]

    log(f"  Protected (val+test) : {len(protected_paths)} images")
    log(f"  Train to check       : {len(train_paths)} images")

    if not protected_paths or not train_paths:
        log("  [EMBED] Nothing to compare — skipping.")
        return hash_clean

    log("  Loading ResNet-18 for embedding extraction…")
    emb_model = _get_embedding_model()

    log("  Extracting val+test embeddings…")
    prot_embs = _extract_embeddings(
        protected_paths, emb_model, desc="  Val+Test embs")

    log("  Extracting train embeddings…")
    train_embs = _extract_embeddings(
        train_paths, emb_model, desc="  Train embs")

    # Chunked cosine similarity: (chunk, embed) × (embed, N_prot) → (chunk, N_prot)
    log("  Computing cosine similarities…")
    flagged   = set()
    sim_pairs = []
    chunk     = 128

    for i in range(0, len(train_embs), chunk):
        batch         = train_embs[i : i + chunk]       # (B, D)
        sims          = torch.mm(batch, prot_embs.T)    # (B, P)
        max_sims, max_idxs = sims.max(dim=1)
        for j, (s, idx) in enumerate(zip(max_sims, max_idxs)):
            if s.item() >= SIM_THRESHOLD:
                gi = i + j
                flagged.add(gi)
                sim_pairs.append({
                    "train"      : train_paths[gi],
                    "matches"    : protected_paths[idx.item()],
                    "similarity" : s.item(),
                })

    # Sort by similarity descending for the report
    sim_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    if sim_pairs:
        log(f"  [EMBED] Found {len(sim_pairs)} near-duplicate pairs "
            f"above threshold={SIM_THRESHOLD}")
        log(f"  [EMBED] Top-5 examples:")
        for p in sim_pairs[:5]:
            log(f"    sim={p['similarity']:.4f} | "
                f"{Path(p['train']).name} ↔ "
                f"{Path(p['matches']).name}")

        # Save similarity pairs report (top 100)
        rep = LOG_DIR / "similar_pairs_report.txt"
        with open(rep, "w") as f:
            f.write(f"Near-duplicate pairs removed from train set\n")
            f.write(f"Cosine-similarity threshold : {SIM_THRESHOLD}\n")
            f.write(f"Total removed               : {len(sim_pairs)}\n")
            f.write("=" * 60 + "\n\n")
            for k, pair in enumerate(sim_pairs[:100], 1):
                f.write(f"{k:3d}.  sim={pair['similarity']:.4f}\n")
                f.write(f"      TRAIN : {pair['train']}\n")
                f.write(f"      MATCH : {pair['matches']}\n\n")
        gdrive_upload(rep, subfolder="logs")

        # Visual grid of most similar pairs
        _visualize_similar_pairs(sim_pairs, top_n=12)
    else:
        log(f"  [EMBED] ✓ No near-duplicates found "
            f"(all similarities < {SIM_THRESHOLD})")

    # Build clean train set
    emb_clean = [(p, l) for idx, (p, l) in enumerate(hash_clean)
                 if idx not in flagged]

    # Free GPU memory used by embedding model
    del emb_model, train_embs, prot_embs
    torch.cuda.empty_cache()

    # Combined report
    total_removed = len(hash_removed) + len(flagged)
    rp = LOG_DIR / "leakage_fix_report.txt"
    with open(rp, "w") as f:
        f.write("LEAKAGE FIX REPORT\n" + "=" * 60 + "\n\n")
        f.write(f"Stage 1 — hash (exact)    : {len(hash_removed)} removed\n")
        f.write(f"Stage 2 — embed (near-dup): {len(flagged)} removed\n")
        f.write(f"Total removed             : {total_removed}\n\n")
        if hash_removed:
            f.write("EXACT DUPLICATES:\n")
            for r in hash_removed:
                f.write(f"  TRAIN : {r['train']}\n")
                f.write(f"  MATCH : {r['matches']}\n\n")
    gdrive_upload(rp, subfolder="logs")

    log(f"\n[LEAKAGE] ─────────────────────────────────────")
    log(f"[LEAKAGE] Stage 1 (exact)     : -{len(hash_removed)}")
    log(f"[LEAKAGE] Stage 2 (near-dup)  : -{len(flagged)}")
    log(f"[LEAKAGE] Total removed       : {total_removed}")
    log(f"[LEAKAGE] Clean train size    : {len(emb_clean)}")

    # Final hash-level verification
    th = {_get_hash(p) for p, _ in emb_clean}
    vh = {_get_hash(p) for p, _ in val_s}
    xh = {_get_hash(p) for p, _ in test_s}
    for s in [th, vh, xh]: s.discard(None)
    overlap = (th & vh) | (th & xh)
    status  = "✓ CLEAN" if not overlap else f"✗ {len(overlap)} hash overlaps remain"
    log(f"[LEAKAGE] Hash verification   : {status}")
    log(f"[LEAKAGE] ─────────────────────────────────────\n")

    return emb_clean


# ─────────────────────────────────────────────────────────────
# OFFLINE AUGMENTATION
# ─────────────────────────────────────────────────────────────
def _augment_pil(img, seed=None):
    if seed is not None: random.seed(seed)
    w,h = img.size
    img = img.rotate(random.uniform(-30,30),
                     resample=Image.BICUBIC,fillcolor=(0,0,0))
    if random.random()>0.5: img=img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random()>0.5: img=img.transpose(Image.FLIP_TOP_BOTTOM)
    img=ImageEnhance.Brightness(img).enhance(random.uniform(0.8,1.2))
    img=ImageEnhance.Contrast(img).enhance(random.uniform(0.8,1.2))
    img=ImageEnhance.Color(img).enhance(random.uniform(0.85,1.15))
    img=ImageEnhance.Sharpness(img).enhance(random.uniform(0.7,1.5))
    if random.random()>0.80:
        img=img.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.2,1.0)))
    if random.random()>0.50:
        r=random.uniform(0.85,1.0); nw=int(w*r); nh=int(h*r)
        l=random.randint(0,w-nw); t=random.randint(0,h-nh)
        img=img.crop((l,t,l+nw,t+nh)).resize((w,h),Image.BICUBIC)
    return img


def generate_augmented_dataset(samples, split_name, n_copies):
    log(f"\n[AUG] {split_name}: {len(samples)} × {n_copies} "
        f"= ~{len(samples)*n_copies}")
    out_dir=AUGMENTED_DIR/split_name

    total_existing=sum(
        len(_collect_images(out_dir/cls))
        for cls in CLASS_NAMES if (out_dir/cls).exists())
    if total_existing>=len(samples)*n_copies*0.9:
        log(f"[AUG] Already done ({total_existing}) — loading.")
        result=[]
        for _,label in samples:
            cls=CLASS_NAMES[label]
            for f in _collect_images(out_dir/cls):
                result.append((str(f),label))
        return result

    result=[]; per_class=defaultdict(list)
    for path,label in samples: per_class[label].append(path)

    for label,paths in per_class.items():
        cls=CLASS_NAMES[label]
        cls_out=out_dir/cls; cls_out.mkdir(parents=True,exist_ok=True)
        count=0
        for img_path in tqdm(paths,
                             desc=f"  [AUG] {split_name}/{cls[:15]}",
                             leave=False):
            try:
                orig=Image.open(img_path).convert("RGB")
                orig=orig.resize((CLF_IMG_SIZE,CLF_IMG_SIZE),Image.BICUBIC)
                for ci in range(n_copies):
                    aug=_augment_pil(orig,seed=hash(img_path)+ci*1000)
                    sp=cls_out/f"{Path(img_path).stem}_aug{ci:02d}.jpg"
                    aug.save(sp,quality=92)
                    result.append((str(sp),label)); count+=1
            except Exception as e:
                log(f"  skip: {e}",also_print=False)
        log(f"  [AUG] {cls:<25} → {count}")
    log(f"[AUG] Total {split_name}: {len(result)}")
    return result


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# (same base for train and val — equal difficulty)
# ─────────────────────────────────────────────────────────────
def get_transforms(split="train"):
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    base=[
        transforms.Resize(int(CLF_IMG_SIZE*1.1),
                          interpolation=Image.BICUBIC),
        transforms.RandomCrop(CLF_IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=25,
            interpolation=transforms.InterpolationMode.BICUBIC,
            fill=0),
        transforms.ColorJitter(
            brightness=0.2,contrast=0.2,
            saturation=0.15,hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ]
    # Train gets very mild erasing only — not enough to suppress acc
    if split=="train":
        return transforms.Compose(base+[
            transforms.RandomErasing(
                p=0.15,scale=(0.02,0.08),ratio=(0.3,3.3))])
    return transforms.Compose(base)

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(int(CLF_IMG_SIZE*1.1),
                          interpolation=Image.BICUBIC),
        transforms.CenterCrop(CLF_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class LeafDataset(Dataset):
    def __init__(self,samples,transform):
        self.samples=samples; self.transform=transform
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        path,label=self.samples[idx]
        try:
            img=Image.open(path).convert("RGB")
        except Exception:
            img=Image.new("RGB",(CLF_IMG_SIZE,CLF_IMG_SIZE),0)
        if self.transform: img=self.transform(img)
        return img,label


# ─────────────────────────────────────────────────────────────
# BUILD DATALOADERS
# ─────────────────────────────────────────────────────────────
def build_dataloaders():
    log("\n"+"="*60)
    log("  BUILDING DATASET")
    log("="*60)

    # Real data
    all_real=[]
    for cls in CLASS_NAMES:
        label=CLASS_NAMES.index(cls)
        for img in _collect_images(REAL_DATA/cls):
            all_real.append((str(img),label))
    log(f"\n  Real images total : {len(all_real)}")

    per_class=defaultdict(list)
    for path,label in all_real: per_class[label].append((path,label))

    real_train=[]; real_val=[]; real_test=[]
    for label,items in per_class.items():
        random.shuffle(items); n=len(items)
        n_test=max(1,int(n*TEST_RATIO))
        n_val =max(1,int(n*VAL_RATIO))
        real_test  +=items[:n_test]
        real_val   +=items[n_test:n_test+n_val]
        real_train +=items[n_test+n_val:]

    log(f"  Real (65/20/15) : "
        f"Train={len(real_train)} Val={len(real_val)} "
        f"Test={len(real_test)}")
    log(f"  Per class       : "
        f"Train≈{len(real_train)//NUM_CLASSES} "
        f"Val≈{len(real_val)//NUM_CLASSES} "
        f"Test≈{len(real_test)//NUM_CLASSES}")

    # Synthetic
    syn_train=[]; syn_val=[]; syn_test=[]
    if SYNTHETIC_DATA.exists():
        for cls in CLASS_NAMES:
            label=CLASS_NAMES.index(cls)
            cls_path=SYNTHETIC_DATA/cls
            if not cls_path.exists(): continue
            imgs=_collect_images(cls_path); random.shuffle(imgs)
            n_tr=SYN_TRAIN_PER_CLASS
            n_va=SYN_VAL_PER_CLASS
            n_te=SYN_TEST_PER_CLASS
            for img in imgs[:n_tr]:
                syn_train.append((str(img),label))
            for img in imgs[n_tr:n_tr+n_va]:
                syn_val.append((str(img),label))
            for img in imgs[n_tr+n_va:n_tr+n_va+n_te]:
                syn_test.append((str(img),label))
        log(f"  Synthetic       : "
            f"Train={len(syn_train)} Val={len(syn_val)} "
            f"Test={len(syn_test)}")
    else:
        log(f"  [WARN] Synthetic not found: {SYNTHETIC_DATA}")

    # Combine
    base_train=real_train+syn_train
    base_val  =real_val  +syn_val
    test_real    =real_test
    test_combined=real_test+syn_test

    # Fix leakage — Stage 1 (hash) + Stage 2 (embeddings)
    clean_train=fix_leakage(base_train, base_val, test_combined)
    log(f"\n  After fix: Train={len(clean_train)} "
        f"(-{len(base_train)-len(clean_train)} removed)")

    # Augmentation ×3 both
    aug_train=generate_augmented_dataset(clean_train,"train",TRAIN_AUG_COPIES)
    aug_val  =generate_augmented_dataset(base_val,   "val",  VAL_AUG_COPIES)

    train_s=clean_train+aug_train
    val_s  =base_val   +aug_val

    log(f"\n  FINAL:")
    log(f"    Train          : {len(train_s)}")
    log(f"    Val            : {len(val_s)}")
    log(f"    Test real      : {len(test_real)}")
    log(f"    Test real+syn  : {len(test_combined)}")

    _save_distribution_chart(train_s,val_s,test_real,test_combined)
    _save_dataset_stats(train_s,val_s,test_real,test_combined,
                        len(syn_train),len(syn_val),len(syn_test))

    kw=dict(num_workers=0,pin_memory=True)
    return (
        DataLoader(LeafDataset(train_s,get_transforms("train")),
                   CLF_BATCH,shuffle=True,drop_last=True,**kw),
        DataLoader(LeafDataset(val_s,  get_transforms("val")),
                   CLF_BATCH,shuffle=False,**kw),
        DataLoader(LeafDataset(test_real,    get_test_transform()),
                   CLF_BATCH,shuffle=False,**kw),
        DataLoader(LeafDataset(test_combined,get_test_transform()),
                   CLF_BATCH,shuffle=False,**kw),
    )


def _save_distribution_chart(train_s,val_s,test_r,test_c):
    fig,axes=plt.subplots(1,4,figsize=(22,6))
    fig.suptitle(
        "Dataset Distribution — Mango Leaf Disease\n"
        "65/20/15 + Synthetic | Hash + Embed Leakage Removed | No Mixup",
        fontsize=12,fontweight="bold")
    colors=plt.cm.Set3(np.linspace(0,1,NUM_CLASSES))
    for ax,(name,samples) in zip(axes,[
        ("Train (Real+Syn+Aug×3)", train_s),
        ("Val   (Real+Syn+Aug×3)", val_s),
        ("Test-Real (15%)",        test_r),
        ("Test-Real+Syn",          test_c),
    ]):
        counts  =Counter(l for _,l in samples)
        cls_cnts=[counts.get(i,0) for i in range(NUM_CLASSES)]
        bars=ax.bar(range(NUM_CLASSES),cls_cnts,
                    color=colors,edgecolor="black",linewidth=0.7)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels([c[:10] for c in CLASS_NAMES],
                           rotation=45,ha="right",fontsize=7)
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\n(n={len(samples)})",fontsize=9)
        ax.grid(axis="y",alpha=0.3)
        for b,cnt in zip(bars,cls_cnts):
            ax.text(b.get_x()+b.get_width()/2,
                    b.get_height()+1,str(cnt),
                    ha="center",va="bottom",fontsize=6)
    plt.tight_layout()
    path=Path("/kaggle/working/dataset_distribution.png")
    plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    gdrive_upload(path,subfolder="plots")
    log("[INFO] Distribution chart saved.")


def _save_dataset_stats(train_s,val_s,test_r,test_c,
                        syn_tr,syn_va,syn_te):
    path=LOG_DIR/"dataset_stats.txt"
    with open(path,"w") as f:
        f.write("="*60+"\n")
        f.write("DATASET STATISTICS\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60+"\n\n")
        f.write(f"Model         : {MODEL_NAME} + LayerNorm Head\n")
        f.write(f"Mixup         : DISABLED (was causing gap)\n")
        f.write(f"Leak detect   : hash + ResNet-18 cosine sim ≥ {SIM_THRESHOLD}\n")
        f.write(f"Split         : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}\n")
        f.write(f"Syn train/cls : {SYN_TRAIN_PER_CLASS}\n")
        f.write(f"Syn val/cls   : {SYN_VAL_PER_CLASS}\n")
        f.write(f"Syn test/cls  : {SYN_TEST_PER_CLASS}\n")
        f.write(f"Aug copies    : ×{TRAIN_AUG_COPIES} both\n\n")
        f.write(f"Train         : {len(train_s)}\n")
        f.write(f"Val           : {len(val_s)}\n")
        f.write(f"Test real     : {len(test_r)}\n")
        f.write(f"Test combined : {len(test_c)}\n\n")
        f.write("HYPERPARAMETERS:\n")
        f.write(f"  Epochs      : {CLF_EPOCHS}\n")
        f.write(f"  Batch       : {CLF_BATCH}\n")
        f.write(f"  Phase 1     : ep1-{PHASE1_END} head only\n")
        f.write(f"  Phase 2     : ep{PHASE2_START}-{CLF_EPOCHS} "
                f"top-{PHASE2_BLOCKS} LR={CLF_LR_BACKBONE}\n")
        f.write(f"  LR head     : {CLF_LR_HEAD}\n")
        f.write(f"  LR backbone : {CLF_LR_BACKBONE}\n")
        f.write(f"  WD P1/P2    : {WD_PHASE1}/{WD_PHASE2}\n")
        f.write(f"  Label smooth: {CLF_LABEL_SMOOTH}\n")
        f.write(f"  Drop rate   : {CLF_DROP_RATE}\n")
        f.write(f"  Drop path   : {CLF_DROP_PATH}\n")
    gdrive_upload(path,subfolder="logs")
    log("[INFO] Dataset stats saved.")


# ─────────────────────────────────────────────────────────────
# LAYERNORM HEAD
# ─────────────────────────────────────────────────────────────
class LNClassificationHead(nn.Module):
    """
    LayerNorm classification head.
    LN is mathematically consistent with DeiT's internal LN.
    No batch-size dependency → same behavior train and test.
    """
    def __init__(self,in_features,num_classes,
                 dim1=HEAD_DIM1,dim2=HEAD_DIM2,
                 drop1=HEAD_DROP1,drop2=HEAD_DROP2):
        super().__init__()
        self.head=nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features,dim1),
            nn.LayerNorm(dim1),
            nn.GELU(),
            nn.Dropout(p=drop1),
            nn.Linear(dim1,dim2),
            nn.LayerNorm(dim2),
            nn.GELU(),
            nn.Dropout(p=drop2),
            nn.Linear(dim2,num_classes),
        )
        for m in self.head:
            if isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self,x): return self.head(x)


class DeiTWithLNHead(nn.Module):
    def __init__(self,backbone,head):
        super().__init__()
        self.backbone=backbone; self.head=head
    def forward(self,x):
        return self.head(self.backbone(x))


def build_model(num_classes):
    import gc; gc.collect(); torch.cuda.empty_cache()
    backbone=timm.create_model(
        MODEL_NAME,pretrained=True,num_classes=0,
        drop_rate=CLF_DROP_RATE,drop_path_rate=CLF_DROP_PATH)
    head=LNClassificationHead(EMBED_DIM,num_classes)
    model=DeiTWithLNHead(backbone,head)
    # Phase 1: head only
    for name,param in model.named_parameters():
        param.requires_grad="head" in name
    total=sum(p.numel() for p in model.parameters())/1e6
    tr   =sum(p.numel() for p in model.parameters()
              if p.requires_grad)/1e6
    log(f"\n[INFO] {MODEL_NAME} + LayerNorm Head")
    log(f"[INFO] Total: {total:.1f}M  Trainable: {tr:.3f}M (head only)")
    log(f"[INFO] Head: LN({EMBED_DIM})→Linear({EMBED_DIM}→{HEAD_DIM1})"
        f"→LN→GELU→Drop→Linear({HEAD_DIM1}→{HEAD_DIM2})"
        f"→LN→GELU→Drop→Linear({HEAD_DIM2}→{num_classes})")
    return model.to(DEVICE)


def _start_phase2(model,optimizer):
    blocks =model.backbone.blocks
    total_b=len(blocks); start=total_b-PHASE2_BLOCKS
    new_p  =[]
    for name,param in model.named_parameters():
        for i in range(start,total_b):
            if (f"backbone.blocks.{i}." in name
                    and not param.requires_grad):
                param.requires_grad=True; new_p.append(param); break
    if new_p:
        optimizer.add_param_group({
            "params":new_p,
            "lr"    :CLF_LR_BACKBONE,
            "weight_decay":WD_PHASE2})
    tr=sum(p.numel() for p in model.parameters()
           if p.requires_grad)/1e6
    log(f"  [PHASE 2] Top {PHASE2_BLOCKS} blocks unfrozen → "
        f"{tr:.1f}M trainable")
    log(f"  [PHASE 2] Backbone LR={CLF_LR_BACKBONE} (micro)  "
        f"WD={WD_PHASE2}")


# ─────────────────────────────────────────────────────────────
# CHECKPOINTS
# ─────────────────────────────────────────────────────────────
def save_checkpoint(epoch,model,optimizer,scheduler,
                    scaler,trl,vll,tra,vla,bva):
    path=CHECKPOINT_DIR/"classifier_latest.pth"
    torch.save({"epoch":epoch,"model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "scaler":scaler.state_dict(),
                "train_losses":trl,"val_losses":vll,
                "train_accs":tra,"val_accs":vla,
                "best_val_acc":bva},path)
    if epoch%CLF_SAVE_EVERY==0:
        mile=CHECKPOINT_DIR/f"clf_ep{epoch:03d}.pth"
        shutil.copy2(path,mile); gdrive_upload(mile,subfolder="checkpoints")
    gdrive_upload(path,subfolder="checkpoints")
    log(f"[CKPT] Saved → ep{epoch} ({path.stat().st_size/1e6:.0f} MB)")


def load_checkpoint(model,optimizer,scheduler,scaler):
    path=CHECKPOINT_DIR/"classifier_latest.pth"
    if not path.exists():
        log("[CKPT] Fresh start."); return 0,[],[],[],[],0.0
    ckpt=torch.load(path,map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    ep=ckpt["epoch"]
    log(f"[CKPT] Resumed from ep{ep} ✓")
    return (ep,ckpt["train_losses"],ckpt["val_losses"],
            ckpt["train_accs"],ckpt["val_accs"],
            ckpt["best_val_acc"])


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def train_classifier(model,train_loader,val_loader):
    log("\n"+"="*60)
    log(f"  TRAINING — {MODEL_NAME} + LN Head")
    log(f"  2-Phase | NO Mixup | Micro backbone LR")
    log(f"  Expected gap: <3% throughout all 60 epochs")
    log("="*60)

    criterion=nn.CrossEntropyLoss(label_smoothing=CLF_LABEL_SMOOTH)
    optimizer=optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CLF_LR_HEAD,weight_decay=WD_PHASE1)

    total_steps =CLF_EPOCHS*len(train_loader)
    warmup_steps=CLF_WARMUP_EP*len(train_loader)

    def lr_lam(step):
        if step<warmup_steps: return step/max(1,warmup_steps)
        p=(step-warmup_steps)/max(1,total_steps-warmup_steps)
        return 0.5*(1+np.cos(np.pi*p))

    scheduler=optim.lr_scheduler.LambdaLR(optimizer,lr_lam)
    scaler   =torch.cuda.amp.GradScaler()

    (start_ep,trl,vll,tra,vla,bva)=load_checkpoint(
        model,optimizer,scheduler,scaler)
    if start_ep>=CLF_EPOCHS:
        log("[INFO] Complete."); return model

    bw           =(copy.deepcopy(model.state_dict())
                   if start_ep>0 else None)
    no_improve   =0
    best_val_loss=min(vll) if vll else float("inf")
    p2_started   =start_ep>=PHASE2_START

    log(f"\n  {'Ep':>5} {'TrLoss':>8} {'TrAcc':>7} "
        f"{'VaLoss':>8} {'VaAcc':>7} {'Gap':>6} {'Phase'}")
    log("  "+"-"*60)

    for epoch in range(start_ep+1,CLF_EPOCHS+1):

        if epoch==PHASE2_START and not p2_started:
            log(f"\n  {'='*56}")
            log(f"  PHASE 2 START — ep{PHASE2_START}")
            log(f"  {'='*56}")
            _start_phase2(model,optimizer)
            p2_started=True

        phase="Head" if epoch<PHASE2_START else f"Top-{PHASE2_BLOCKS}"

        # ── Train ─────────────────────────────────────────────
        model.train(); el=ec=et=0
        for imgs,labels in tqdm(
                train_loader,
                desc=f"  ep{epoch:03d}/{CLF_EPOCHS} [{phase}]",
                leave=False):
            imgs  =imgs.to(DEVICE,non_blocking=True)
            labels=labels.to(DEVICE,non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits=model(imgs)
                loss  =criterion(logits,labels)  # NO mixup

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

            el+=loss.item()*imgs.size(0)
            ec+=(logits.argmax(1)==labels).sum().item()
            et+=imgs.size(0)

        train_loss=el/et; train_acc=ec/et

        # ── Validate ──────────────────────────────────────────
        model.eval(); vl=vc=vt=0
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs  =imgs.to(DEVICE,non_blocking=True)
                labels=labels.to(DEVICE,non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits=model(imgs); loss=criterion(logits,labels)
                vl+=loss.item()*imgs.size(0)
                vc+=(logits.argmax(1)==labels).sum().item()
                vt+=imgs.size(0)

        val_loss=vl/vt; val_acc=vc/vt
        gap=abs(val_acc-train_acc)*100

        trl.append(train_loss); vll.append(val_loss)
        tra.append(train_acc);  vla.append(val_acc)

        log(f"  Ep [{epoch:3d}/{CLF_EPOCHS}]  "
            f"Tr:{train_loss:.4f}/{train_acc:.4f}  "
            f"Va:{val_loss:.4f}/{val_acc:.4f}  "
            f"Gap:{gap:5.1f}%  [{phase}]")

        if val_acc>bva:
            bva=val_acc; bw=copy.deepcopy(model.state_dict())
            bp=CHECKPOINT_DIR/"best_deit_ln.pth"
            torch.save(bw,bp); gdrive_upload(bp,subfolder="checkpoints")
            log(f"    ★  Best saved (val={bva:.4f})")

        if val_loss<best_val_loss-1e-4:
            best_val_loss=val_loss; no_improve=0
        else:
            no_improve+=1
            if no_improve>=EARLY_STOP_PATIENCE:
                log(f"\n  [EARLY STOP] {EARLY_STOP_PATIENCE} ep "
                    f"no improvement.")
                save_checkpoint(epoch,model,optimizer,scheduler,
                                scaler,trl,vll,tra,vla,bva); break

        if epoch%CLF_SAVE_EVERY==0 or epoch==CLF_EPOCHS:
            save_checkpoint(epoch,model,optimizer,scheduler,
                            scaler,trl,vll,tra,vla,bva)

    log(f"\n[INFO] Best val acc: {bva:.4f}")
    _save_training_curves(trl,vll,tra,vla)
    if bw is not None: model.load_state_dict(bw)
    return model


def _save_training_curves(trl,vll,tra,vla):
    epochs=list(range(1,len(trl)+1))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(
        "DeiT-Small + LayerNorm Head — Training Curves\n"
        "Mango Leaf Disease | No Mixup | 2-Phase | "
        "Micro Backbone LR",
        fontsize=12,fontweight="bold")

    for ax,(ys_tr,ys_va,ylabel,title) in zip(axes,[
        (trl,vll,"Loss","Loss Curves"),
        ([a*100 for a in tra],[a*100 for a in vla],
         "Accuracy (%)","Accuracy Curves"),
    ]):
        ax.plot(epochs,ys_tr,color="#534AB7",linewidth=2.5,
                marker="o",markersize=2,label="Train")
        ax.plot(epochs,ys_va,color="#E74C3C",linewidth=2.5,
                marker="s",markersize=2,label="Val")
        if "%" in ylabel:
            ax.fill_between(epochs,ys_tr,ys_va,
                            alpha=0.06,color="gray",label="Gap")
        if PHASE2_START<=len(epochs):
            ax.axvline(x=PHASE2_START,color="green",
                       linestyle="--",linewidth=1.5,alpha=0.8,
                       label=f"Phase 2 start (ep{PHASE2_START})")
        ax.set_xlabel("Epoch",fontsize=11)
        ax.set_ylabel(ylabel,fontsize=11)
        ax.set_title(title,fontsize=11)

        # ── Legend position fix ───────────────────────────────
        # Loss curves fall from high→low: legend at lower-right
        # would hide the converged tails. Use upper-right instead.
        # Accuracy curves rise from low→high: lower-right is clear.
        leg_loc = "upper right" if "%" not in ylabel else "lower right"
        ax.legend(fontsize=9, loc=leg_loc)
        # ─────────────────────────────────────────────────────

        ax.grid(True,alpha=0.3)
        if "%" in ylabel: ax.set_ylim(0,100)

    plt.tight_layout()
    path=Path("/kaggle/working/training_curves.png")
    plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    gdrive_upload(path,subfolder="plots")
    log("[INFO] Training curves saved.")


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def _run_evaluation(model,loader,label_name):
    model.eval(); ap=[]; al=[]
    with torch.no_grad():
        for imgs,labels in tqdm(loader,
                                desc=f"  Eval {label_name}",
                                leave=False):
            imgs=imgs.to(DEVICE,non_blocking=True)
            with torch.cuda.amp.autocast():
                out=model(imgs)
            ap.append(out.argmax(1).cpu()); al.append(labels)
    preds=torch.cat(ap); labels=torch.cat(al); n=NUM_CLASSES
    f1s =MulticlassF1Score(num_classes=n,average=None)(preds,labels).numpy()
    prec=MulticlassPrecision(num_classes=n,average=None)(preds,labels).numpy()
    rec =MulticlassRecall(num_classes=n,average=None)(preds,labels).numpy()
    cm  =MulticlassConfusionMatrix(num_classes=n)(preds,labels).numpy()
    acc =(preds==labels).float().mean().item()
    return acc,float(f1s.mean()),float(prec.mean()),float(rec.mean()),\
           f1s,prec,rec,cm


def evaluate(model,test_real_loader,test_comb_loader):
    log("\n"+"="*60)
    log("  EVALUATION — DETAILED RESULTS")
    log("="*60)

    results={}
    for tag,loader in [("Real Only",test_real_loader),
                       ("Real+Synthetic",test_comb_loader)]:
        log(f"\n  ── {tag} ──")
        (acc,fm,pm,rm,f1s,prec,rec,cm)=_run_evaluation(
            model,loader,tag)
        results[tag]=(acc,fm,pm,rm,f1s,prec,rec,cm)

        log(f"\n  {'Metric':<25} {'Value':>10}")
        log(f"  {'-'*37}")
        log(f"  {'Overall Accuracy':<25} {acc:>10.4f}  ({acc*100:.2f}%)")
        log(f"  {'Macro F1 Score':<25} {fm:>10.4f}")
        log(f"  {'Macro Precision':<25} {pm:>10.4f}")
        log(f"  {'Macro Recall':<25} {rm:>10.4f}")
        log(f"\n  {'Class':<35} {'F1':>7} {'Prec':>7} {'Rec':>7}")
        log(f"  {'-'*60}")
        for cls,f1,pr,rc in zip(CLASS_NAMES,f1s,prec,rec):
            log(f"  {cls:<35} {f1:>7.4f} {pr:>7.4f} {rc:>7.4f}  "
                f"{'█'*int(f1*20)}")

    # Summary
    (acc_r,fm_r,pm_r,rm_r,f1s_r,prec_r,rec_r,cm_r)=results["Real Only"]
    (acc_c,fm_c,pm_c,rm_c,f1s_c,prec_c,rec_c,cm_c)=results["Real+Synthetic"]

    log(f"\n  ── SUMMARY ──")
    log(f"  {'Metric':<30} {'Real Only':>12} {'Real+Syn':>12}")
    log(f"  {'-'*56}")
    log(f"  {'Accuracy':<30} {acc_r:>12.4f} {acc_c:>12.4f}")
    log(f"  {'Macro F1':<30} {fm_r:>12.4f} {fm_c:>12.4f}")
    log(f"  {'Precision':<30} {pm_r:>12.4f} {pm_c:>12.4f}")
    log(f"  {'Recall':<30} {rm_r:>12.4f} {rm_c:>12.4f}")

    # Plots
    _plot_f1_comparison(f1s_r,f1s_c,fm_r,fm_c)
    for data,fmt,title,fname in [
        (cm_r,"d","Confusion Matrix — Real Test Only",
         "confusion_matrix_real.png"),
        (cm_r.astype(float)/(cm_r.sum(1,keepdims=True)+1e-9),
         ".2f","Normalized CM — Real Only",
         "confusion_matrix_real_norm.png"),
        (cm_c,"d","Confusion Matrix — Real+Synthetic Test",
         "confusion_matrix_combined.png"),
        (cm_c.astype(float)/(cm_c.sum(1,keepdims=True)+1e-9),
         ".2f","Normalized CM — Real+Synthetic",
         "confusion_matrix_combined_norm.png"),
    ]:
        _plot_confusion(data,fmt,title,fname)

    # Save text report
    rp=LOG_DIR/"evaluation_results.txt"
    with open(rp,"w") as f:
        f.write("="*60+"\n"); f.write("EVALUATION RESULTS\n")
        f.write(f"Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model : {MODEL_NAME} + LayerNorm Head\n")
        f.write("="*60+"\n\n")
        for tag,(acc,fm,pm,rm,f1s,_,_,_) in results.items():
            f.write(f"TEST: {tag}\n")
            f.write(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)\n")
            f.write(f"  Macro F1  : {fm:.4f}\n")
            f.write(f"  Precision : {pm:.4f}\n")
            f.write(f"  Recall    : {rm:.4f}\n\n")
            f.write(f"  {'Class':<35} {'F1':>10}\n")
            f.write("  "+"-"*47+"\n")
            for cls,s in zip(CLASS_NAMES,f1s):
                f.write(f"  {cls:<35} {s:>10.4f}\n")
            f.write("\n")
    gdrive_upload(rp,subfolder="logs")
    log("[INFO] All evaluation outputs saved.")
    return f1s_r,cm_r


def _plot_f1_comparison(f1s_r,f1s_c,fm_r,fm_c):
    x=np.arange(NUM_CLASSES); w=0.35
    fig,ax=plt.subplots(figsize=(14,6))
    fig.suptitle(
        "Per-Class F1 — Real vs Real+Synthetic Test\n"
        "DeiT-Small + LN Head | Mango Leaf Disease",
        fontsize=12,fontweight="bold")
    b1=ax.bar(x-w/2,f1s_r,w,label=f"Real only (F1={fm_r:.4f})",
              color="#534AB7",edgecolor="black",linewidth=0.7)
    b2=ax.bar(x+w/2,f1s_c,w,label=f"Real+Syn  (F1={fm_c:.4f})",
              color="#E74C3C",edgecolor="black",linewidth=0.7,alpha=0.8)
    ax.axhline(fm_r,color="#534AB7",linestyle="--",linewidth=1.5,alpha=0.6)
    ax.axhline(fm_c,color="#E74C3C",linestyle="--",linewidth=1.5,alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES,rotation=40,ha="right",fontsize=10)
    ax.set_ylim(0,1.12); ax.set_ylabel("F1 Score",fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.3)
    for b,s in zip(list(b1)+list(b2),list(f1s_r)+list(f1s_c)):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.012,
                f"{s:.3f}",ha="center",va="bottom",fontsize=7,
                fontweight="bold")
    plt.tight_layout()
    path=Path("/kaggle/working/f1_comparison.png")
    plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    gdrive_upload(path,subfolder="plots")


def _plot_confusion(data,fmt,title,fname):
    fig,ax=plt.subplots(figsize=(13,11))
    sns.heatmap(data,annot=True,fmt=fmt,cmap="Blues",
                xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,
                linewidths=0.5,linecolor="lightgray",
                annot_kws={"size":10},ax=ax)
    ax.set_xlabel("Predicted",fontsize=12)
    ax.set_ylabel("True",fontsize=12)
    ax.set_title(
        f"{title}\nDeiT-Small + LN Head | Mango Leaf Disease",
        fontsize=12,fontweight="bold")
    plt.xticks(rotation=40,ha="right",fontsize=9)
    plt.yticks(rotation=0,fontsize=9); plt.tight_layout()
    p=Path(f"/kaggle/working/{fname}")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    gdrive_upload(p,subfolder="plots")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    start=time.time()
    log("\n"+"█"*60)
    log(f"  MANGO — DeiT-Small + LN Head (NO MIXUP)")
    log(f"  Leakage: Hash + ResNet-18 embedding cosine sim")
    log(f"  Diagnosis: Mixup was sole cause of 10-13% gap")
    log(f"  Fix: Removed Mixup → expected gap <3% all epochs")
    log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("█"*60+"\n")

    setup_gdrive()

    (train_loader,val_loader,
     test_real_loader,
     test_comb_loader)=build_dataloaders()

    model=build_model(NUM_CLASSES)
    model=train_classifier(model,train_loader,val_loader)

    # ── Save best model for download ──────────────────────────
    # train_classifier restores the best weights before returning,
    # so model.state_dict() here IS the best model.
    final_save = Path("/kaggle/working/mango_deit_ln_best.pth")
    torch.save(
        {
            # Weights
            "model_state_dict": model.state_dict(),
            # Reconstruction metadata
            "model_name"      : MODEL_NAME,
            "class_names"     : CLASS_NAMES,
            "num_classes"     : NUM_CLASSES,
            "img_size"        : CLF_IMG_SIZE,
            "embed_dim"       : EMBED_DIM,
            "head_dim1"       : HEAD_DIM1,
            "head_dim2"       : HEAD_DIM2,
            "head_drop1"      : HEAD_DROP1,
            "head_drop2"      : HEAD_DROP2,
            "drop_rate"       : CLF_DROP_RATE,
            "drop_path"       : CLF_DROP_PATH,
        },
        final_save,
    )
    sz = final_save.stat().st_size / 1e6
    log(f"\n[SAVE] ══════════════════════════════════════════")
    log(f"[SAVE] ★  Best model saved for download:")
    log(f"[SAVE]    → /kaggle/working/mango_deit_ln_best.pth")
    log(f"[SAVE]    Size : {sz:.1f} MB")
    log(f"[SAVE]    Keys : model_state_dict, class_names, ...")
    log(f"[SAVE] ══════════════════════════════════════════")
    gdrive_upload(final_save,subfolder="checkpoints")

    evaluate(model,test_real_loader,test_comb_loader)
    gdrive_upload(LOG_FILE,subfolder="logs")

    elapsed=(time.time()-start)/3600
    log(f"\n[INFO] Total time: {elapsed:.2f} hrs")
    log("  ★  DONE — Model at /kaggle/working/mango_deit_ln_best.pth")
    log("  ★  Check Google Drive for all plots and logs!")


if __name__ == "__main__":
    main()