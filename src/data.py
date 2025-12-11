import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GermanNewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    # def __getitem__(self, idx):
    #     text = str(self.texts[idx])
        
    #     encoding = self.tokenizer(
    #         text,
    #         truncation=True,
    #         padding=False, 
    #         max_length=self.max_len,
    #         return_token_type_ids=False
    #     )
        
    #     item = {
    #         'input_ids': encoding['input_ids'],
    #         'attention_mask': encoding['attention_mask']
    #     }
        
    #     if self.labels is not None:
    #         item['labels'] = self.labels[idx]
            
    #     return item
    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Отримуємо весь текст без обрізки
        encoding = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_token_type_ids=False
        )
        input_ids = encoding["input_ids"]

        max_len = self.max_len  # 512
        if len(input_ids) > max_len:
            half = max_len // 2  # 256
            head = input_ids[:half]
            tail = input_ids[-half:]
            input_ids = head + tail

        attention_mask = [1] * len(input_ids)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item

class DataHandler:
    def __init__(self, train_path, test_path, tokenizer, max_len=512, seed=42):
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.id_to_label = None

    def load_csv(self, path):
        texts, labels = [], []
        seen = set()
        
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "label" in row: 
                    if row["text"] not in seen:
                        texts.append(row["text"])
                        labels.append(row["label"])
                        seen.add(row["text"])
                else: 
                    texts.append(row["text"])
                    labels.append(row["id"])
        return texts, labels

    def get_collate_fn(self):
        pad_token_id = self.tokenizer.pad_token_id or 0
        fixed_len = self.max_len  # 512

        def collate_fn(batch):
            input_ids = []
            attention_masks = []
            labels = []

            for x in batch:
                ids = x["input_ids"]
                mask = x["attention_mask"]

                # на всякий випадок, щоб точно не було > fixed_len
                if len(ids) > fixed_len:
                    ids = ids[:fixed_len]
                    mask = mask[:fixed_len]

                pad_len = fixed_len - len(ids)

                input_ids.append(ids + [pad_token_id] * pad_len)
                attention_masks.append(mask + [0] * pad_len)

                if "labels" in x:
                    labels.append(x["labels"])

            batch_dict = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
            if labels:
                batch_dict["labels"] = torch.tensor(labels, dtype=torch.long)

            return batch_dict

        return collate_fn


    def get_dataloaders(self, batch_size=16):
        texts, str_labels = self.load_csv(self.train_path)
        
        unique = sorted(set(str_labels))
        lbl2id = {l: i for i, l in enumerate(unique)}
        self.id_to_label = {i: l for i, l in enumerate(unique)}
        int_labels = [lbl2id[l] for l in str_labels]
        
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(int_labels), y=int_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            texts, int_labels, test_size=0.1, stratify=int_labels, random_state=self.seed
        )
        
        train_ds = GermanNewsDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_ds = GermanNewsDataset(X_val, y_val, self.tokenizer, self.max_len)
        
        collate = self.get_collate_fn()
        kwargs = {
            "num_workers": 4, 
            "pin_memory": True, 
            "persistent_workers": True
        }
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, **kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, **kwargs)

        return train_loader, val_loader, weights # Returning weights so Trainer can use them

    def get_test_loader(self, batch_size=16):
        texts, ids = self.load_csv(self.test_path)
        kwargs = {
            "num_workers": 4, 
            "pin_memory": True, 
            "persistent_workers": True
        }
        test_ds = GermanNewsDataset(texts, labels=None, tokenizer=self.tokenizer, max_len=self.max_len)

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=self.get_collate_fn(), **kwargs)

        return test_loader, ids
    
    def get_full_train_loader(self, batch_size=16):
        texts, str_labels = self.load_csv(self.train_path)

        unique = sorted(set(str_labels))
        lbl2id = {l: i for i, l in enumerate(unique)}
        self.id_to_label = {i: l for i, l in enumerate(unique)}
        int_labels = [lbl2id[l] for l in str_labels]

        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(int_labels), y=int_labels
        )

        train_ds = GermanNewsDataset(texts, int_labels, self.tokenizer, self.max_len)

        collate = self.get_collate_fn()
        kwargs = {
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
        }

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            **kwargs,
        )

        return train_loader, weights