#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
download_data.py
iNaturalistのAPIを使って、東京エリアの植物(Plantae)観察から画像URLを取得し、
data/raw/ 配下にダウンロードするスクリプト

※ 70,000件のデータ取得を目標とし、日付範囲でクエリを分割してデータを取得します。
※ ダウンロード対象の進捗状況（取得済み画像数と全体に対するパーセンテージ）も表示します。
"""

import os
import time
import requests
from datetime import datetime, timedelta
from pyinaturalist import get_observations

# 東京近辺にざっくり合わせたバウンディングボックス (例)
SWLAT, SWLNG = 34.58, 138.30  # 南西端
NELAT, NELNG = 36.44, 141.24  # 北東端

# iNaturalist上での植物界 taxon_id (Plantae)
TAXON_ID = 47126

# データ保存先ディレクトリ
RAW_DIR = os.path.join("data", "raw")

# 一度のAPI呼び出しで取得する最大件数
PER_PAGE = 50

# 1ブロックあたりのページ数
BLOCK_SIZE = 100

# 目標取得件数
TARGET_COUNT = 70000

# 日付範囲設定（例: 2010-01-01から2023-12-31）
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2023, 12, 31)
# 日付範囲の刻み（例: 7日ごと）
DELTA = timedelta(days=7)

def fetch_data_for_period(d1, d2):
    block_observations = []
    page = 1
    while page <= BLOCK_SIZE:
        print(f"Fetching page {page} for period {d1.strftime('%Y-%m-%d')} to {d2.strftime('%Y-%m-%d')} ...")
        try:
            response = get_observations(
                taxon_id=TAXON_ID,
                swlat=SWLAT, swlng=SWLNG,
                nelat=NELAT, nelng=NELNG,
                quality_grade='research',
                per_page=PER_PAGE,
                page=page,
                d1=d1.strftime('%Y-%m-%d'),
                d2=d2.strftime('%Y-%m-%d')
            )
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError on page {page} for period {d1.strftime('%Y-%m-%d')} to {d2.strftime('%Y-%m-%d')}: {e}")
            break

        results = response.get('results', [])
        if not results:
            print("No more observations found in this period.")
            break

        block_observations.extend(results)
        time.sleep(1)
        page += 1

    print(f"Completed period {d1.strftime('%Y-%m-%d')} to {d2.strftime('%Y-%m-%d')}: Fetched {len(block_observations)} observations.")
    return block_observations

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    all_observations = []
    current_start = START_DATE

    # 日付範囲を7日ごとに分割して、TARGET_COUNT件までデータを取得
    while current_start <= END_DATE and len(all_observations) < TARGET_COUNT:
        current_end = current_start + DELTA
        if current_end > END_DATE:
            current_end = END_DATE

        print(f"\n=== Fetching data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} ===")
        observations = fetch_data_for_period(current_start, current_end)
        all_observations.extend(observations)
        print(f"Total observations so far: {len(all_observations)}")
        if len(all_observations) >= TARGET_COUNT:
            print(f"Reached target of {TARGET_COUNT} observations.")
            break

        current_start = current_end + timedelta(days=1)

    print(f"\nTotal fetched observations: {len(all_observations)}")

    # ダウンロード対象（写真がある観察）の合計数を先に計算
    total_downloadable = sum(1 for obs in all_observations if obs.get('photos'))
    downloaded_count = 0

    # 画像をダウンロード
    for obs in all_observations:
        photos = obs.get('photos', [])
        if not photos:
            continue

        obs_id = obs['id']
        taxon_name = obs.get('species_guess') or "Unknown"
        # 複数写真がある場合は1枚目のみ取得
        photo_info = photos[0]
        image_url = photo_info.get('url') or photo_info.get('original_url')
        if image_url:
            safe_taxon_name = taxon_name.replace(" ", "_").replace("/", "_")
            filename = f"{obs_id}_{safe_taxon_name}.jpg"
            save_path = os.path.join(RAW_DIR, filename)
            if not os.path.exists(save_path):
                try:
                    r = requests.get(image_url, timeout=10)
                    if r.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(r.content)
                        print(f"Downloaded {filename}")
                    else:
                        print(f"Failed to get {image_url} (status: {r.status_code})")
                except Exception as e:
                    print(f"Error downloading {image_url}: {e}")
            else:
                print(f"Already exists: {filename}")

            downloaded_count += 1
            progress_percentage = (downloaded_count / total_downloadable) * 100 if total_downloadable > 0 else 0
            print(f"Progress: {downloaded_count}/{total_downloadable} images downloaded, {progress_percentage:.2f}% complete.\n")

    print("Download completed.")

if __name__ == "__main__":
    main()
