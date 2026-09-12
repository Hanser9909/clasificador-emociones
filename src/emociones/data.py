import numpy as np
import pandas as pd
import torch
import yaml

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

MAX_LEN = 64

def cargar_config():

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def cargar_datos(config):
    """Carga el dataset de emociones."""

    dataset = load_dataset(config["data"]["dataset"])

    class_names = dataset["train"].features["label"].names
    num_classes = len(class_names)

    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()

    return (
        df_train,
        df_val,
        df_test,
        class_names,
        num_classes,
    )


def tokenizar_datos(df_train, df_val, df_test, config):
    """Tokeniza los textos utilizando BERT."""

    tokenizer = BertTokenizer.from_pretrained(
        config["model"]["name"],
        use_fast=True
    )

    def tokenizar(textos):
        enc = tokenizer(
            list(textos),
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return enc["input_ids"], enc["attention_mask"]

    train_ids, train_mask = tokenizar(df_train["text"].values)
    val_ids, val_mask = tokenizar(df_val["text"].values)
    test_ids, test_mask = tokenizar(df_test["text"].values)

    y_train = df_train["label"].values
    y_val = df_val["label"].values
    y_test = df_test["label"].values

    train_ds = TensorDataset(
        train_ids,
        train_mask,
        torch.tensor(y_train, dtype=torch.long)
    )

    val_ds = TensorDataset(
        val_ids,
        val_mask,
        torch.tensor(y_val, dtype=torch.long)
    )

    test_ds = TensorDataset(
        test_ids,
        test_mask,
        torch.tensor(y_test, dtype=torch.long)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    return train_dl, val_dl, test_dl, tokenizer