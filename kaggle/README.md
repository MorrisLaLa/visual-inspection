# Kaggle 視覺檢測程式（Ultralytics YOLO26）

在 Kaggle 免費 GPU 上訓練與評估 YOLO 檢測模型，分成兩個領域：

| 檔案 | 用途 |
|---|---|
| `medical_yolo.py` | **醫學影像**：DICOM、CSV 框標註（VinDr-CXR / RSNA）、分割遮罩（Kvasir-SEG / ISIC）、分類資料夾；依病人分組切分 |
| `semiconductor_yolo.py` | **工業 / 半導體**：大圖小缺陷自動切塊（PCB、AOI、SEM）、VOC / COCO / YOLO、WM-811K 晶圓圖分類；依批號分組切分 |
| `vi_core.py` | 共用核心：格式轉換、分組切分、訓練、調門檻、評估、上線關卡、複判佇列、匯出 |
| `notebooks/` | 可直接匯入 Kaggle 的 Notebook 範本 |
| `tests/` | 單元測試（不需要 GPU）：`pytest kaggle/tests` |

## 每次執行會做什麼

1. **讀資料**：自動判斷格式（YOLO yaml / 資料夾、Pascal VOC XML、COCO JSON、分類資料夾、CSV、遮罩、WM-811K pickle）。
2. **分組切分 train / val / test**：同一位病人或同一批號（lot）的影像一定在同一組，避免資料洩漏。切分依「分組 ID 的雜湊值」決定，之後加新資料時**舊影像的分組不會變**，所以測試集可以長期固定，當作黃金測試集。
3. **大圖切塊**（半導體預設自動判斷）：影像長邊超過模型輸入的 1.5 倍時，切成重疊的小塊來訓練。這樣小缺陷不會因為整張縮小而消失。
4. **訓練 YOLO26**：預設用 `yolo26s`，並依領域套用不同的資料增強。醫學預設不左右翻轉；晶圓圖不做任何亮度增強。
5. **在 val 上逐類調門檻**：每一類取「仍能達到 `--target-recall` 的最高門檻」，限制在 `--min-threshold` ～ `--max-threshold` 之間。
6. **在 test 上用同一組門檻報告**：在**整張原圖**上評估，切塊模型也會先切塊推論再合併，跟實際上線時一樣。
7. **上線關卡**（`--baseline`）：新舊模型在同一份 test 上比較。漏檢不能變多、誤報不能變多、mAP 或準確率不能明顯下降。
8. **複判佇列**（`--unlabeled`）：挑出模型最沒把握的影像，再加 10% 隨機抽樣，附上 YOLO 格式預標註，打包成 zip 給人工複判。
9. **匯出**：預設輸出 ONNX；另可選 `engine`（TensorRT）、`openvino`、`tflite`、`coreml`。

## 在 Kaggle 上執行

1. **新建 Notebook，設定環境：** 右側 **Settings** 中，Accelerator 選 **GPU T4 x2** 或 **P100**，**Internet** 打開。打開 Internet 需要先完成手機驗證，下載 YOLO 權重和 ultralytics 都要用到網路。
2. **取得程式**，以下方法擇一：
   - **A：** 把本 repo 的 `kaggle/` 資料夾上傳成 Kaggle Dataset（Datasets → New Dataset），再到 Notebook 用 **Add Data** 加入。
   - **B：** repo 是公開的話，在 Notebook 裡執行 `!git clone --depth 1 https://github.com/MorrisLaLa/visual-inspection.git`。
   - **C：** 直接匯入 `notebooks/medical_kaggle.ipynb` 或 `notebooks/semiconductor_kaggle.ipynb`（File → Import Notebook）。這兩個範本已包含 A、B 兩種取得程式的方式。
3. **加入資料：** 用 **Add Data** 加入你要訓練的資料集。
4. **執行**（路徑依你加入的資料集為準）：

```bash
!python /kaggle/input/<程式資料集>/semiconductor_yolo.py --data /kaggle/input/<你的資料集>/...
```

輸出會放在 `/kaggle/working/medical/` 或 `/kaggle/working/semiconductor/`，執行完可以從右側 Output 下載。

## 醫學影像範例

```bash
# 示範：Ultralytics 內建 brain-tumor 資料集（自動下載，不需要 Add Data）
!python medical_yolo.py --epochs 50

# VinDr-CXR 胸部 X 光（DICOM + CSV）：多位醫師的重複框合併、CLAHE、縮到 1024 以節省空間
!python medical_yolo.py --csv /kaggle/input/<vindr>/train.csv --images /kaggle/input/<vindr>/train \
    --merge-iou 0.4 --clahe --max-size 1024 --imgsz 1024

# RSNA Pneumonia（patientId, x, y, width, height, Target；空白框 = 正常影像）
!python medical_yolo.py --csv /kaggle/input/<rsna>/stage_2_train_labels.csv \
    --images /kaggle/input/<rsna>/stage_2_train_images --image-col patientId --patient-col patientId \
    --class-name pneumonia --max-size 1024 --imgsz 1024

# Kvasir-SEG 息肉分割（影像 + 遮罩 → YOLO 分割）
!python medical_yolo.py --data /kaggle/input/<kvasir>/Kvasir-SEG/images \
    --masks /kaggle/input/<kvasir>/Kvasir-SEG/masks --mask-classes polyp

# ISIC 皮膚病灶分割（遮罩檔名有 _segmentation 後綴）
!python medical_yolo.py --data <影像資料夾> --masks <遮罩資料夾> --mask-suffix _segmentation

# Chest X-Ray Pneumonia 分類：原本的 val 只有 16 張，所以重切；用檔名裡的病人編號分組
!python medical_yolo.py --data /kaggle/input/<chest-xray>/chest_xray --resplit \
    --group-regex '^(person\d+|NORMAL2-IM-\d+|IM-\d+)'
```

- **DICOM 窗位：** 預設套用 DICOM 內建的 Modality LUT 和 VOI LUT，再依 0.5–99.5% 百分位正規化。要指定窗位窗寬時用 `--window <中心> <寬度>`（例如肺窗 `--window -600 1500`）。
- **MONOCHROME1** 會自動反相。
- **壓縮的 DICOM**（JPEG2000 / JPEG-LS）第一次讀取失敗時，會自動安裝 `pylibjpeg` 再試一次。
- **轉檔快取：** 轉好的 PNG 放在 `/tmp/vi-datasets/medical_cache/`，同一個 session 重跑不會再轉一次。
- **縮小版 PNG 資料集：** 如果改用別人縮小過的 PNG 版本，CSV 的框座標也必須是縮小後的座標。

## 工業 / 半導體範例

```bash
# PCB Defects（Pascal VOC XML，約 3000px 大圖、6 類小缺陷）→ 自動切塊
!python semiconductor_yolo.py --data /kaggle/input/<pcb-defects>/PCB_DATASET --imgsz 640

# WM-811K 晶圓圖（LSWMD.pkl）：缺陷圖樣分類，依 lotName 分組；未標註晶圓自動進複判佇列
!python semiconductor_yolo.py --data /kaggle/input/<wm811k>/LSWMD.pkl --wafer-max-per-class 3000

# 自有 AOI / SEM 影像（YOLO 或 COCO 格式），用檔名裡的批號分組
!python semiconductor_yolo.py --data /kaggle/input/<my-aoi>/ --group-regex '^(LOT\w+)_'

# 分類資料夾（例如 good/、scratch/、particle/），good 會自動當作良品
!python semiconductor_yolo.py --data /kaggle/input/<my-cls>/
```

- **切塊大小：** 預設等於 `--imgsz`，重疊 20%。缺陷尺寸要小於重疊寬度（640 × 0.2 = 128 px），才保證每個缺陷都能完整出現在某一塊裡。
- **WM-811K 舊版 pickle：** 這個檔案是用很舊的 pandas 存的，程式會自動處理新版 pandas 讀不到 `pandas.indexes` 的問題。不過本 repo 只用格式相同的合成 pickle 測試過，**還沒在真正的 LSWMD.pkl 上跑過**；如果讀取失敗，請回報錯誤訊息。
- **16-bit TIFF：** Ultralytics 讀取時會轉成 8-bit。原圖對比很低時，建議先自行正規化成 8-bit。

## 輸出檔案

| 檔案 | 說明 |
|---|---|
| `report.md` / `report.json` | 主要指標、每類門檻與召回率、上線關卡結果 |
| `best.pt` | 最佳權重（下一輪用 `--model` 接續訓練、用 `--baseline` 比較） |
| `best.onnx` | 部署用模型 |
| `thresholds.json` | 上線用的每類門檻（分類任務則是「正常」類別與異常門檻） |
| `split_manifest.csv` | 每張影像的分組與 split，也就是黃金測試集的清單 |
| `train/` | 訓練曲線、混淆矩陣、PR 曲線 |
| `val_test/` | Ultralytics 在 test 上的評估圖表 |
| `review_queue/` / `review_queue.zip` | 待複判影像、YOLO 預標註、`queue.csv`（挑選原因、離門檻多近） |

## 閉環：每一輪怎麼做

1. **第 1 輪訓練：**

   ```bash
   !python semiconductor_yolo.py --data <資料> --unlabeled <未標註影像>
   ```

   跑完後下載 `best.pt` 和 `review_queue.zip`。
2. **人工複判：** 把 `review_queue/`（YOLO 格式，含 `data.yaml`、`classes.txt`）匯入 CVAT 或 Label Studio，只修正預標註就好。
3. **更新資料集：** 把修好的標註併入原資料集，上傳成 Kaggle Dataset 的新版本。
4. **第 2 輪：** 從上一輪的模型接續訓練，並和它比較：

   ```bash
   !python semiconductor_yolo.py --data <新版資料> \
       --model /kaggle/input/<上一輪輸出>/best.pt \
       --baseline /kaggle/input/<上一輪輸出>/best.pt \
       --unlabeled <新一批未標註影像>
   ```

5. **決定要不要換模型：** 看 `report.md` 最後的「上線關卡」。通過才換，接著先跑影子模式（旁路比對、不控制產線），再灰度上線。

切分依分組雜湊決定，舊影像的 split 每一輪都相同，新舊模型永遠在同一份測試集上比較。

> **例外：** 資料量很少時，程式會從 train 移一兩組到 val/test，確保每個類別和無缺陷影像都有樣本。這種微調可能隨新資料改變分組；資料量夠大時不會觸發。

## 指標說明

| 指標 | 意義 |
|---|---|
| 漏檢率 / 漏診率 | 標註的缺陷或病灶中，沒被抓到（IoU ≥ `--match-iou` 且同類別）的比例。產線最重要的指標 |
| 過殺率 / 偽陽性率 | 無缺陷或正常影像被判成 NG / 異常的比例；test 裡要有這類影像才算得出來 |
| 誤報數 / 每張影像 | 沒有對應到任何標註的預測框數量；test 沒有正常影像時用它代替過殺率 |
| 敏感度 / 特異度 / AUC | 整張影像層級，或分類任務「異常 vs 正常」的二元判斷 |
| mAP50 / mAP50-95 | Ultralytics 標準指標（切塊模型是在 test 切塊上計算） |

**門檻怎麼調出來的：**
- 在 val 上，每一類取「仍能達到 `--target-recall`（醫學預設 95%、半導體預設 98%）的最高門檻」，上限是 `--max-threshold`（預設 0.5）。
- 設上限的原因：val 樣本少時，「剛好達標的最高門檻」很容易過度貼合 val，到 test 反而漏掉。
- 下限是 `--min-threshold`（預設 0.1）：達到目標召回率需要比下限更低的門檻時，門檻停在下限，不會為了召回率一路降到誤報暴增。
- 報告顯示「⚠️ 達不到目標（停在下限 0.1）」時，代表在合理的門檻內，val 上達不到目標召回率，報告裡的召回率就是停在下限時的實際值。這時需要更多資料、更多訓練、更大的 `--imgsz`；或者確認可以接受較多誤報後，自己調低 `--min-threshold`。

## 常用參數

| 參數 | 說明 |
|---|---|
| `--size n/s/m/l/x` | 模型大小（預設 s） |
| `--imgsz` | 模型輸入尺寸（偵測預設 640、分類 224、晶圓 128） |
| `--epochs` / `--time` | 訓練輪數 / 時數上限（Kaggle 單次 session 最長約 12 小時） |
| `--batch` | 預設 -1，依 GPU 記憶體自動決定 |
| `--task` | auto / detect / segment / classify（例如分割資料只想做偵測時用 detect） |
| `--target-recall` / `--min-threshold` / `--max-threshold` | 調門檻的目標召回率，以及門檻下限（預設 0.1）與上限（預設 0.5） |
| `--tune-level object/image` | object：每個缺陷都要抓到；image：只看整張影像判 OK/NG |
| `--group-regex` / `--patient-col` | 從檔名 / CSV 欄位取分組 ID，避免資料洩漏 |
| `--tile on/off/auto` | 大圖切塊（只用於 detect） |
| `--resplit` | 忽略來源的 train/val/test，依分組重切 |
| `--set key=value` | 覆蓋任何 Ultralytics 訓練參數，例如 `--set lr0=0.005 mosaic=0.5 fraction=0.2` |
| `--skip-train --model best.pt` | 不訓練，只評估或產生複判佇列 |
| `--export` | onnx（預設）/ engine / openvino / tflite / coreml / none |

## 注意事項

- **醫學用途：** 結果僅供研究與教學，不可作為臨床診斷依據。真正上線需要獨立的臨床驗證與法規程序。
- **3D 影像：** NIfTI 等 3D 體積影像目前不支援，請先轉成 2D 切片影像。DICOM 多影格只取中間那張。
- **Kaggle 限制：**
  - GPU 有每週時數配額，單次 session 最長約 12 小時，可用 `--time` 控制訓練時數。
  - `/kaggle/working` 輸出上限 20GB，所以轉換後的資料集放在 `/tmp`，不會佔用輸出空間。
- **授權：** Ultralytics YOLO（含 YOLO26）是 AGPL-3.0。閉源商用請購買 Ultralytics 企業授權，或請法務確認。
