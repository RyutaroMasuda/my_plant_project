#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
realsense_inference_pil.py
RealSense D435 で取得した映像を PyTorch の学習済みモデルで推論し、
PILでテキストを描画してOpenCVで表示するデモスクリプト。
日本語ラベルを表示するため、cv2.putText() は使わず、PIL.ImageDraw.text() を利用。
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import pyrealsense2 as rs

from PIL import Image, ImageDraw, ImageFont  # PIL系のモジュールをインポート

# 学習時と同じ画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# クラス名のリスト（日本語含む）
CLASS_NAMES = [
    'Allium_tuberosum', 'Arisaema_angustatum', 'Beach_Vitex', 'Bunch-flowered_Daffodil', 'Camphor', 'Camphora_officinarum', 'Candelabra_Aloe', 'Chameleon_Plant', 'China_knotweed', 'Chinaberry', 'Chinese_silver_grass', 'Chusan_Palm', 'Eurasian_water-milfoil', 'Fatsia_japonica', 'Formosa_lily', 'Fringed_Iris', 'Giant_Butterbur', 'Glabrous_Sarcandra_Herb', 'Green_penny_fern', 'Heavenly_bamboo', 'Japanese_Kerria', 'Japanese_Persimmon', 'Japanese_aralia', 'Japanese_aucuba', 'Japanese_camellia', 'Japanese_cedar', 'Japanese_cheesewood', 'Japanese_maple', 'Japanese_snake_gourd', 'Japanese_zelkova', 'Kudzu_Bean', 'Marvel_of_Peru', 'Miscanthus_sinensis', 'Pacific_Chrysanthemum', 'Paper_mulberry', 'Sacred_Lotus', 'Trichosanthes_cucumeroides', 'Weeping_Fern', 'Yellow_Cosmos', "bird's-eye_speedwell", 'cutleaf_evening_primrose', 'ginkgo', 'henbit_deadnettle', 'kadsura_vine', 'kudzu', 'leopard-plant', 'low_smartweed', 'pink_knotweed', "raisin_d'Amérique", 'rose_evening_primrose', 'skunk_vine', 'sugi', 'tall_goldenrod', 'Каркас_китайский', 'アオキ', 'アオギリ', 'アオツヅラフジ', 'アオハダ', 'アオミズ', 'アカシデ', 'アカツメクサ', 'アカネ', 'アカマツ', 'アカメガシワ', 'アキグミ(広義)', 'アキニレ', 'アキノエノコログサ', 'アキノタムラソウ', 'アキノノゲシ', 'アケビ', 'アシボソ(広義)', 'アズマイバラ', 'アセビ', 'アブラススキ', 'アメリカアサガオ', 'アメリカオニアザミ', 'アラカシ', 'アレチウリ', 'アレチヌスビトハギ', 'アレチハナガサ', 'イイギリ', 'イシミカワ', 'イソギク', 'イタドリ', 'イチョウ', 'イヌガラシ', 'イヌコウジュ', 'イヌシデ', 'イヌタデ', 'イヌツゲ', 'イヌホオズキ', 'イヌワラビ', 'イノコヅチ', 'イノモトソウ', 'イモカタバミ', 'イロハモミジ', 'イワニガナ', 'ウグイスカグラ(広義)', 'ウシハコベ', 'ウスベニニガナ', 'ウチワゼニクサ', 'ウバユリ', 'ウメモドキ', 'ウラシマソウ', 'エゴノキ', 'エノキ', 'エノキグサ', 'エノコログサ', 'エビヅル', 'エビモ', 'オオイヌノフグリ', 'オオカナダモ', 'オオカワヂシャ', 'オオジシバリ', 'オオニシキソウ', 'オオハナワラビ', 'オオバギボウシ', 'オオバコ', 'オオバナノセンダングサ', 'オオバノイノモトソウ', 'オオフサモ', 'オオブタクサ', 'オキナダケ', 'オギ', 'オクマワラビ', 'オケラ', 'オシロイバナ', 'オニグルミ', 'オニタビラコ', 'オニドコロ', 'オヒシバ', 'オランダカイウ', 'カキ', 'カクレミノ', 'カシワバハグマ', 'カジノキ', 'カタバミ', 'カナムグラ', 'カニクサ', 'カマツカ', 'カヤツリグサ', 'カヤラン', 'カラスウリ', 'カラスザンショウ', 'カラスノゴマ', 'カラムシ', 'カラムシ（広義）', 'カワラナデシコ', 'カワラノギク', 'カンツバキ', 'ガガイモ', 'ガマズミ', 'キクザキリュウキンカ', 'キチジョウソウ', 'キツネササゲ', 'キツネノマゴ', 'キツネノマゴ(広義)', 'キヅタ', 'キバナアキギリ', 'キバナコスモス', 'キブシ', 'キョウチクトウ', 'キンエノコロ', 'ギンゴケ', 'クサイチゴ', 'クサギ', 'クサノオウ', 'クスノキ', 'クズ_（広義）', 'クズ_（狭義）', 'クヌギ', 'クマシデ', 'クマヤナギ', 'クモノスシダ', 'クリ', 'クロガネモチ', 'クワクサ', 'ケゼニゴケ', 'ケヤキ', 'コウゾリナ', 'コウヤボウキ', 'コゴメウツギ', 'コセンダングサ(広義)', 'コナギ', 'コナスビ', 'コナラ', 'コニシキソウ', 'コブナグサ', 'コマツヨイグサ', 'コムラサキ', 'ゴンズイ', 'サザンカ', 'サネカズラ', 'サルトリイバラ', 'サンカクカタバミ', 'サンショウ', 'ザクロ', 'シオデ', 'シナダレスズメガヤ', 'シノブ', 'シマスズメノヒエ', 'シモバシラ', 'シュロ', 'シュロソウ', 'ショウジョウソウ', 'シラカシ', 'シラヤマギク', 'シロダモ', 'シロツメクサ', 'シロバナタンポポ', 'シロヤマブキ', 'シロヨメナ', 'ジュウモンジシダ', 'ジュズダマ', 'スイカズラ', 'スイセン', 'スカシタゴボウ', 'スギ', 'スギナ', 'ススキ', 'スズメウリ', 'スダジイ', 'セイタカアワダチソウ', 'セイバンモロコシ', 'セイヨウタンポポ', 'センダン', 'センニンソウ', 'センリョウ', 'ゼニゴケ', 'ゼンマイ', 'ソクズ', 'タケニグサ', 'タチツボスミレ', 'タマガヤツリ', 'タマサンゴ', 'タマスダレ', 'タマノカンアオイ', 'ダンドボロギク', 'チカラシバ', 'チヂミザサ', 'チャノキ', 'チョウジタデ', 'ツタ', 'ツユクサ', 'ツリフネソウ', 'ツルウメモドキ', 'ツルソバ', 'ツルナ', 'ツルニチニチソウ', 'ツルボ', 'ツルマンネングサ', 'ツワブキ', 'テイカカズラ', 'トウネズミモチ', 'トウバナ', 'トキリマメ', 'トキワサンザシ', 'トキワツユクサ', 'トキワハゼ', 'トダシバ', 'トネアザミ', 'トベラ', 'トラノオシダ', 'ドクダミ', 'ナガエコミカンソウ', 'ナガバヤブソテツ', 'ナガミヒナゲシ', 'ナキリスゲ', 'ナズナ', 'ナワシロイチゴ', 'ナワシログミ', 'ナンキンハゼ', 'ナンテン', 'ニシキギ', 'ニセアカシア', 'ニラ', 'ニワウルシ', 'ヌカキビ', 'ヌスビトハギ', 'ヌルデ', 'ネコハギ', 'ネジバナ', 'ネズミノオ', 'ネズミモチ', 'ネムノキ', 'ノアサガオ', 'ノイバラ', 'ノガリヤス', 'ノキシノブ', 'ノゲシ', 'ノコンギク', 'ノダケ', 'ノビル', 'ノブドウ', 'ハキダメギク', 'ハゼノキ', 'ハゼラン', 'ハナウリクサ', 'ハナタデ', 'ハナニラ', 'ハナヤエムグラ', 'ハハコグサ', 'ハマヒルガオ', 'ハマボッス', 'ハリギリ', 'ハルシャギク', 'ハルジオン', 'ハンショウヅル', 'ヒイラギ', 'ヒイラギナンテン', 'ヒガンバナ', 'ヒサカキ', 'ヒデリコ', 'ヒノキ', 'ヒメウズ', 'ヒメオドリコソウ', 'ヒメコウゾ', 'ヒメジャゴケ', 'ヒメジョオン', 'ヒメツルソバ', 'ヒヨドリジョウゴ', 'ヒヨドリバナ', 'ヒロハフウリンホオズキ', 'ビロードシダ', 'ビロードモウズイカ', 'ビワ', 'フイリゲンジスミレ', 'フウセンカズラ', 'フキ', 'フクジュソウ', 'フジ', 'フジバカマ', 'フモトシダ', 'フユイチゴ', 'フユノハナワラビ', 'ブタナ', 'ヘクソカズラ', 'ヘビイチゴ', 'ヘラオオバコ', 'ベニシダ', 'ベニバナサルビア', 'ベニバナボロギク', 'ペラペラヨメナ', 'ホウライシダ', 'ホオノキ', 'ホザキノフサモ', 'ホシダ', 'ホタルブクロ', 'ホテイアオイ', 'ホトケノザ', 'マキバチカラシバ', 'マスクサ', 'マテバシイ', 'ママコノシリヌグイ', 'マメヅタ', 'マユミ', 'マルバアサガオ', 'マルバハッカ', 'マルバルコウ', 'マンリョウ', 'ミズキ', 'ミズヒキ', 'ミゾソバ', 'ミツデウラボシ', 'ミツバアケビ', 'ミドリヒメワラビ', 'ミヤコグサ', 'ミヤコヤブソテツ', 'ムクノキ', 'ムベ', 'ムラサキカタバミ', 'ムラサキシキブ', 'メリケンカルカヤ', 'メリケンガヤツリ', 'モミジイチゴ', 'モミジバフウ', 'ヤクシソウ', 'ヤツデ', 'ヤドリギ', 'ヤハズエンドウ(広義)', 'ヤハズソウ', 'ヤブカラシ', 'ヤブコウジ', 'ヤブソテツ', 'ヤブツバキ', 'ヤブマメ', 'ヤブミョウガ', 'ヤブムラサキ', 'ヤブラン', 'ヤブレガサ', 'ヤマウコギ', 'ヤマグワ', 'ヤマノイモ', 'ヤマハギ', 'ヤマハッカ', 'ヤマブキ', 'ヤマホトトギス', 'ヤマユリ', 'ユウゲショウ', 'ユキノシタ', 'ユズリハ', 'ユリノキ', 'ヨウシュヤマゴボウ', 'ヨシ', 'ヨモギ', 'ランタナ', 'リュウノウギク', 'リュウノヒゲ', 'リョウメンシダ', 'リンドウ', 'ロウバイ', 'ワラビ', 'ワルナスビ', 'ワレモコウ', '八角金盤', '北美一枝黃花', '北美鵝掌楸', '圓錐鐵線蓮', '寶蓋草', '山菊', '樟', '美洲商陸', '芒', '虎杖', '銀杏', '鐃鈸花', '鐵線蕨', '雞屎藤', '雞爪槭', '頭花蓼', '魚腥草'
]

def load_model(model_path, num_classes):
    """ResNet18に学習済み重みを読み込み"""
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # クラス数とCLASS_NAMESが合っていることを確認
    model = load_model("plant_classifier.pth", num_classes=len(CLASS_NAMES))
    model.to(device)
    model.eval()

    # RealSenseパイプラインの開始
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("RealSense pipeline started. Press ESC to exit.")

    # **日本語フォントファイル** を指定
    # 環境に合わせて適切なフォント(.ttfや.ttc)を指定してください
    # 例: ipa-gothic, NotoSansCJK, TakaoGothic, MSMinchoなど
    font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"  # 例
    font = ImageFont.truetype(font_path, size=32)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # OpenCV用の BGR NumPy配列
            color_image = np.asanyarray(color_frame.get_data())

            # NumPy(BGR) → RGB → PIL.Image
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # 推論用 前処理 (PIL → Tensor)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # 推論
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                pred_label = predicted.item()

            # クラス名を取得
            if 0 <= pred_label < len(CLASS_NAMES):
                pred_class = CLASS_NAMES[pred_label]
            else:
                pred_class = "Unknown"

            # PILで日本語含むテキストを描画
            draw = ImageDraw.Draw(pil_image)
            text = f"Prediction: {pred_class}"
            # (10,10)に 緑色（RGB=(0,255,0)）で描画
            draw.text((10, 10), text, fill=(0, 255, 0), font=font)

            # PIL → NumPy(BGR) に再変換してOpenCV表示
            annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            cv2.imshow('RealSense', annotated_image)

            if cv2.waitKey(1) == 27:  # ESCキー
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
