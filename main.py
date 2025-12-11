# full_train.py

import argparse
import os
import random
import numpy as np
import torch

from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.custom import GermanClassifier, get_tokenizer
from src.data import DataHandler, GermanNewsDataset
from src.trainer import train_one_epoch, get_llrd_params


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_full_train_loader(handler: DataHandler, batch_size: int):
    """
    Лоадер на ВСЬОМУ train.csv (без train/val split).
    Логіка така сама, як у handler.get_dataloaders, тільки без split.
    """
    # 1. Завантажуємо весь train
    texts, str_labels = handler.load_csv(handler.train_path)

    # 2. Мапа label -> id
    unique = sorted(set(str_labels))
    lbl2id = {l: i for i, l in enumerate(unique)}
    handler.id_to_label = {i: l for i, l in enumerate(unique)}
    int_labels = [lbl2id[l] for l in str_labels]

    # 3. Ваги класів
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(int_labels),
        y=int_labels,
    )

    # 4. Dataset
    train_ds = GermanNewsDataset(
        texts=texts,
        labels=int_labels,
        tokenizer=handler.tokenizer,
        max_len=handler.max_len,
    )

    collate = handler.get_collate_fn()
    kwargs = {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        **kwargs,
    )

    return loader, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepset/gbert-large")
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--max_len", type=int, default=512)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--peak_lr", type=float, default=2e-5)
    parser.add_argument("--layer_decay", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--accumulation_steps", type=int, default=1)

    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    set_seed(args.seed)

    # Tokenizer + DataHandler
    tokenizer = get_tokenizer(args.model_name)
    handler = DataHandler(
        train_path=args.train_path,
        test_path=args.test_path,
        tokenizer=tokenizer,
        max_len=args.max_len,
        seed=args.seed,
    )

    train_loader, class_weights = get_full_train_loader(handler, args.batch_size)

    # Модель
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    model = GermanClassifier(
        model_name=args.model_name,
        num_labels=len(handler.id_to_label),
        class_weights=class_weights_t,
        dropout_rate=0.1,
    ).to(device)

    # Оптимізатор з LLRD (беремо існуючу реалізацію з trainer.py)
    optimizer_grouped_parameters = get_llrd_params(
        model=model,
        peak_lr=args.peak_lr,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Scheduler
    from transformers import get_linear_schedule_with_warmup

    total_steps = (len(train_loader) // args.accumulation_steps) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP scaler
    from torch.amp import GradScaler

    scaler = GradScaler()

    # Опційно — torch.compile, як у ноутбуку
    try:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="default")
    except Exception as e:
        print("torch.compile failed, using eager mode:", e)

    print(
        f"Start full-train: model={args.model_name}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}"
    )

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
        )
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f}"
        )

    # Зберігаємо фінальну модель (розкомпільовану, якщо треба)
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    if args.save_path is None:
        os.makedirs("saved_models", exist_ok=True)
        fname = (
            f"fulltrain_{os.path.basename(args.model_name).replace('/', '_')}"
            f"_seed{args.seed}_ep{args.epochs}.pth"
        )
        args.save_path = os.path.join("saved_models", fname)

    torch.save(save_model.state_dict(), args.save_path)
    print("Saved model to", args.save_path)


if __name__ == "__main__":
    main()
