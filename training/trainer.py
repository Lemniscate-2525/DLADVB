import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from config.config import (
    DEVICE,
    TRAIN_FILE,
    VAL_FILE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    OUTPUT_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
)

from training.dataset import LogDataset
from training.evaluate import evaluate
from models.bert_classifier import build_model


def train():

    train_dataset = LogDataset(TRAIN_FILE)
    val_dataset = LogDataset(VAL_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = build_model().to(DEVICE)

    optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_start = time.time()

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0
        correct = 0
        total = 0

        start = time.time()

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = epoch_loss / len(train_loader)

        precision, recall, f1, val_acc = evaluate(
            model,
            val_loader
        )

        print("=" * 60)

        print(f"Epoch {epoch+1}/{EPOCHS}")

        print(f"Train Loss      : {avg_loss:.4f}")

        print(f"Train Accuracy  : {train_acc:.2f}%")

        print(f"Val Accuracy    : {val_acc:.2f}%")

        print(f"Precision       : {precision:.4f}")

        print(f"Recall          : {recall:.4f}")

        print(f"F1 Score        : {f1:.4f}")

        print(f"Epoch Time      : {time.time()-start:.2f}s")

        if f1 > best_f1:

            best_f1 = f1

            model.save_pretrained(OUTPUT_DIR)

            train_dataset.tokenizer.save_pretrained(
                OUTPUT_DIR
            )

            print("Best model saved.")

    print("=" * 60)

    print(f"Training Finished.")

    print(f"Total Time : {(time.time()-total_start)/60:.2f} Minutes")


if __name__ == "__main__":
    train()