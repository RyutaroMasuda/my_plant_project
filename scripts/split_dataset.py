#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
split_dataset_filtered.py

data/raw/ にあるファイルを、 species_name ごとのフォルダに振り分け、
Train データと Test データに分割する。
ただし、いずれかのタイミングでファイル数が0枚になったクラスは除外する(作成しない)。
"""

import os
import shutil
import random
from glob import glob

RAW_DIR = os.path.join("data", "raw")
TRAIN_DIR = os.path.join("data", "train")
TEST_DIR = os.path.join("data", "test")

# テストデータに回す割合
TEST_RATIO = 0.2

def main():
    # ディレクトリを作成
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # rawフォルダ内の全画像ファイルパスを取得（拡張子jpgのみと仮定）
    image_paths = glob(os.path.join(RAW_DIR, "*.jpg"))

    # 種名(フォルダ名) -> 画像パスリスト でグループ化
    species_dict = {}
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        # 例: "1234567_Acer_palmatum.jpg" →  "Acer_palmatum" を種名として扱う
        parts = filename.split("_", 1)
        if len(parts) > 1:
            taxon_name = parts[1].rsplit(".", 1)[0]  # 拡張子除去
        else:
            taxon_name = "Unknown"
        
        species_dict.setdefault(taxon_name, []).append(img_path)

    # 各種ごとにランダムに分割
    for taxon_name, paths in species_dict.items():
        # パスが0枚ならスキップ
        if len(paths) == 0:
            print(f"Skipping {taxon_name}: 0 images.")
            continue

        # シャッフル
        random.shuffle(paths)

        # 分割インデックス
        split_idx = int(len(paths) * (1 - TEST_RATIO))
        # 念のため split_idx が 0 or len(paths) にならないようチェック
        # → 例えば画像が1枚の場合、test_ratio=0.2でも切り捨て=0になる
        if split_idx == 0:
            # train に1枚も行かないので、train=0, test=全枚数 という状態
            # この場合は学習に使えないのでスキップ
            print(f"Skipping {taxon_name}: all images would go to test (total={len(paths)})")
            continue
        if split_idx >= len(paths):
            # test に1枚も行かない(全てtrain)の可能性があるが、学習自体は問題なし
            # → 気になるなら split_idx = len(paths) - 1 など調整しても良い
            pass
        
        train_paths = paths[:split_idx]
        test_paths  = paths[split_idx:]

        # 分割後、train_paths が 0枚ならスキップ
        if len(train_paths) == 0:
            print(f"Skipping {taxon_name}: train set is 0 images after split.")
            continue

        # ディレクトリ作成
        train_species_dir = os.path.join(TRAIN_DIR, taxon_name)
        test_species_dir  = os.path.join(TEST_DIR, taxon_name)
        os.makedirs(train_species_dir, exist_ok=True)
        os.makedirs(test_species_dir, exist_ok=True)

        # コピー (または shutil.move なども可)
        for p in train_paths:
            shutil.copy2(p, train_species_dir)
        for p in test_paths:
            shutil.copy2(p, test_species_dir)

        print(f"{taxon_name}: {len(train_paths)} train, {len(test_paths)} test")

    print("Dataset split completed with filtering.")


if __name__ == "__main__":
    main()
