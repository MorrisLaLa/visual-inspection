# visual-inspection
Visual inspection using smartphone and CCTV cameras - 手機/監視鏡頭視覺檢測

用 YOLO 模型辨識 **車輛、人員、動物**，可以讀取以下來源：

- 電腦的 webcam
- 手機鏡頭
- 監視器（RTSP 串流）
- 錄好的影片或照片

這三類都在 COCO 資料集的類別裡，現成的預訓練模型直接就能辨識，**不需要先訓練**。等實際畫面辨識效果不夠好時（例如夜間紅外線畫面、特殊角度、當地特有動物），再用自己的資料微調就好。

| 分類 | 包含的 COCO 類別 |
|---|---|
| `vehicle` 車輛 | bicycle, car, motorcycle, bus, truck, train, boat, airplane |
| `person` 人員 | person |
| `animal` 動物 | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |

分類規則在 [`visual_inspection/categories.py`](visual_inspection/categories.py)，要增減類別就改這裡。

## 安裝

需要 Python 3.10 以上。

```bash
pip install -r requirements.txt
```

第一次執行時會自動下載模型權重（`yolo26n.pt` 約 5 MB）。

## 使用方式

```bash
# 電腦的 webcam
python -m visual_inspection.detect --source 0

# 手機鏡頭（見下方「手機鏡頭」）
python -m visual_inspection.detect --source http://192.168.1.20:8080/video

# 監視器 RTSP 串流，加上追蹤，每個物體只算一次
python -m visual_inspection.detect --source "rtsp://帳號:密碼@192.168.1.10:554/Streaming/Channels/102" --track

# 影片：存下標好框的影片，並把每筆偵測寫進 CSV
python -m visual_inspection.detect --source clip.mp4 --save --log detections.csv

# 整個資料夾的照片，只找動物
python -m visual_inspection.detect --source photos/ --categories animal --save
```

畫面視窗按 `q` 結束。照片模式下，按任意鍵會換下一張。

### 常用參數

| 參數 | 說明 |
|---|---|
| `--source` | 影像來源：`0`（webcam）、RTSP/HTTP 網址、影片檔、照片或照片資料夾 |
| `--model` | 模型權重，預設 `yolo26n.pt`。越大越準也越慢：`yolo26n` → `s` → `m` → `l` → `x`；也可以用自己訓練的 `best.pt` |
| `--categories` | 只偵測某幾類，例如 `--categories vehicle person` |
| `--conf` | 信心度門檻，預設 0.4。誤報多就調高，漏抓多就調低 |
| `--track` | 跨畫面追蹤物體，結束時列出各類「不重複」的數量 |
| `--stride N` | 影片每 N 幀才處理一幀，處理長影片時比較快 |
| `--imgsz` | 推論的影像大小，預設 640。電腦較慢時可以改成 480 或 320 |
| `--device` | `cpu`、`0`（第一張 NVIDIA 顯卡）、`mps`（Apple 晶片的 Mac），預設自動選 |
| `--save` | 把標好框的照片或影片存到 `outputs/`（可用 `--save-dir` 改位置） |
| `--log CSV` | 把每一筆偵測（時間、類別、信心度、座標、追蹤編號）附加寫入 CSV |
| `--no-show` | 不開預覽視窗，在伺服器或背景執行時使用 |

### 手機鏡頭

手機和電腦要連同一個 Wi-Fi，在手機上安裝可以把鏡頭變成網路攝影機的 App：

- **Android**：IP Webcam，網址通常是 `http://手機IP:8080/video`
- **Android / iPhone**：DroidCam，網址通常是 `http://手機IP:4747/video`

實際網址以 App 畫面上顯示的為準。

### 監視器（CCTV）

大部分網路監視器都支援 RTSP。網址格式依品牌不同，常見的有：

| 品牌 | 主碼流（高畫質） | 子碼流（低畫質、較快） |
|---|---|---|
| Hikvision 海康 | `rtsp://帳號:密碼@IP:554/Streaming/Channels/101` | `.../Channels/102` |
| Dahua 大華 | `rtsp://帳號:密碼@IP:554/cam/realmonitor?channel=1&subtype=0` | `...&subtype=1` |

其他品牌可以查說明書，或用 ONVIF Device Manager 找出網址。

建議：

- **用子碼流**：辨識不需要 4K 畫面，解析度低反而跑得快。
- **斷線會自動重連**：Wi-Fi 監視器常會短暫斷線，程式會自動重新連線，不會直接結束。
- **畫面不會越來越延遲**：電腦處理速度跟不上鏡頭時，程式只處理最新一幀，不會越積越多。
- ⚠️ **密碼不要寫進程式碼或提交到 git。**

### 速度參考

只有 CPU 的電腦，建議用 `yolo26n` 搭配子碼流，必要時加上 `--imgsz 480`。有 NVIDIA 顯卡可以改用 `yolo26s` 或 `yolo26m`，會更準。

## 用 Kaggle 的免費 GPU 測試和訓練

[`kaggle/yolo_kaggle.ipynb`](kaggle/yolo_kaggle.ipynb) 會做兩件事：

1. **比較模型**：在你上傳的照片和影片上跑 `yolo26n` / `s` / `m`，比較速度和偵測數量，並輸出標好框的結果。
2. **訓練（選用）**：有自己標註的資料集時，在 GPU 上微調，產出 `best.pt`。

### 方法一：在 Kaggle 網頁上執行

1. 帳號要先完成手機驗證，才能使用 GPU 和網路。
2. 把影片或照片上傳成 Kaggle Dataset，設為 **Private**。監視器畫面可能拍到人臉和車牌，請勿公開。
3. 新增 Notebook → File → Import Notebook，選 `kaggle/yolo_kaggle.ipynb`。
4. 右側 Settings：
   - Accelerator 選 **GPU T4 x2** 或 **GPU P100**
   - Internet 打開
   - Add Input 加入剛才的 Dataset
5. 用 **Save Version → Save & Run All** 執行。完成後到 Output 下載結果。

### 方法二：用 Kaggle 指令（適合讓 Claude Code 幫你操作）

1. 到 Kaggle → Settings → API → **Create New Token**，把下載的 `kaggle.json` 放到 `~/.kaggle/`（Windows：`C:\Users\你的帳號\.kaggle\`）。
2. 修改 `kaggle/kernel-metadata.json`：
   - 把 `YOUR_KAGGLE_USERNAME` 換成你的帳號
   - 在 `dataset_sources` 填入你的 Dataset，例如 `"你的帳號/my-cctv-clips"`

```bash
pip install kaggle

# 上傳資料（預設就是私人）
kaggle datasets init -p my-cctv-clips      # 產生 dataset-metadata.json，編輯裡面的 title 和 id
kaggle datasets create -p my-cctv-clips

# 把筆記本送到 Kaggle GPU 執行，查看狀態，下載結果
kaggle kernels push -p kaggle
kaggle kernels status 你的帳號/visual-inspection-yolo
kaggle kernels output 你的帳號/visual-inspection-yolo -p outputs/kaggle
```

### 用自己的資料訓練

1. 用 Label Studio、CVAT 或 Roboflow 標註照片，匯出成 **YOLO 格式**。
2. 參考 [`data/dataset.yaml`](data/dataset.yaml) 整理資料夾，把 `dataset.yaml` 放在資料集根目錄。
3. 上傳成 Kaggle Dataset，執行筆記本，它會自動找到 `dataset.yaml` 開始訓練。
4. 下載 `best.pt`，在本機使用：

   ```bash
   python -m visual_inspection.detect --source 0 --model best.pt
   ```

## 專案結構

```
visual_inspection/
  categories.py     類別 → 車輛/人員/動物 的對應
  detect.py         偵測主程式（攝影機、影片、照片）
kaggle/
  yolo_kaggle.ipynb    Kaggle GPU 測試與訓練筆記本
  kernel-metadata.json 用 kaggle 指令上傳時的設定
data/
  dataset.yaml      自訂訓練資料集範本
tests/              單元測試（不需要下載模型）
```

## 測試

```bash
pip install pytest
python -m pytest
```

## 授權注意

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 採用 AGPL-3.0 授權。如果要用在商業產品中、又不想公開原始碼，需要向 Ultralytics 購買 Enterprise 授權。
