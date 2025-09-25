# coco_to_classification_and_train.py
from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_ROOT = Path("data_basket")   
CLS_ROOT     = Path("cls_ds")    # dataset de classification que l'on va générer
TARGET_CLASS_KEYWORDS = ["basket"]  # on détecte la catégorie par nom (ex: "basketball")

def find_annotation_json(dir_path: Path) -> Path:
    jsons = list(dir_path.glob("*.json")) + list((dir_path/"annotations").glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"Aucun JSON trouvé dans {dir_path}")
    return jsons[0]

def load_coco(dir_path: Path) -> Dict:
    with find_annotation_json(dir_path).open("r", encoding="utf-8") as f:
        return json.load(f)

def get_target_category_ids(coco: Dict) -> List[int]:
    ids = []
    for c in coco["categories"]:
        name = c["name"].lower()
        if any(k in name for k in TARGET_CLASS_KEYWORDS):
            ids.append(c["id"])
    if not ids:
        # fallback : si une seule catégorie dans le COCO, on la prend
        if len(coco["categories"]) == 1:
            ids = [coco["categories"][0]["id"]]
        else:
            raise RuntimeError("Impossible de trouver la catégorie 'basketball' (ajuste TARGET_CLASS_KEYWORDS).")
    return ids

def build_index(coco: Dict):
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    return images_by_id, anns_by_img

def make_classification_split(split_name: str):
    split_dir = DATASET_ROOT / split_name
    coco = load_coco(split_dir)
    target_cat_ids = set(get_target_category_ids(coco))
    images_by_id, anns_by_img = build_index(coco)

    out_pos = CLS_ROOT / split_name / "basketball"
    out_neg = CLS_ROOT / split_name / "other"
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        src = split_dir / file_name
        anns = anns_by_img.get(img_id, [])
        has_target = any(a["category_id"] in target_cat_ids for a in anns)
        dst_dir = out_pos if has_target else out_neg
        shutil.copy2(src, dst_dir / Path(file_name).name)
        copied += 1
    print(f"[{split_name}] copiées: {copied} images -> {out_pos} / {out_neg}")

def build_keras_datasets(img_size=(160,160), batch_size=32):
    def make(split):
        return tf.keras.preprocessing.image_dataset_from_directory(
            CLS_ROOT / split,
            labels="inferred",
            label_mode="binary",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=42,
        ).prefetch(tf.data.AUTOTUNE)
    return make("train"), make("valid"), make("test")

def build_classifier(input_shape=(160,160,3)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    # Transfer learning léger (MobileNetV2)
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False  # on commence en gelant la base
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, base

if __name__ == "__main__":
    # 1) Construire le dataset de classification à partir du COCO
    for split in ["train", "valid", "test"]:
        make_classification_split(split)

    # 2) Charger pour Keras
    train_ds, val_ds, test_ds = build_keras_datasets(img_size=(160,160), batch_size=32)

    # 3) Modèle + entraînement
    model, base = build_classifier((160,160,3))
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    # 4) Débloquer une partie de la base pour fine-tuning (optionnel)
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    # 5) Évaluation
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
