# 相機實測工具（先用 iPhone，之後換監視鏡頭或工業相機）

`camtest.py` 用 Kaggle 訓練好的 YOLO 模型，實際測試相機在產線速度下能不能檢：

1. **先算**：還沒架相機就能算出哪些速度可行。
2. **再錄影測**：錄下不同速度的影片，和答案比對，得到「**最高可用速度**」。
3. **最後即時跑**：接上相機即時檢測。

每個步驟都有防呆，拿不準的工件一律判「**需人工確認**」，**絕不放行成良品**。

換相機時流程完全相同：只改設定檔的 `camera` 段，然後重新 `calib`。

```
Kaggle 訓練（best.pt、thresholds.json）
   │
   ▼
plan  ──►  架設 iPhone  ──►  calib（放尺）  ──►  錄影 + test（速度測試）  ──►  live（即時）
                                                        │
                                                        ▼
                                  parts/NG、CHECK、OK 影像 → 人工確認 → 加進 Kaggle 資料集重訓
```

## 安裝

在 Mac（建議，可以直接用 iPhone 當攝影機）、Windows 或 Linux 電腦上：

```bash
pip install -r camera_test/requirements.txt
cp camera_test/config.example.yaml config.yaml    # 之後只改這個檔
```

把 Kaggle 輸出的 `best.pt`、`thresholds.json` 放到設定檔 `inspect.model`、`inspect.thresholds` 指定的位置。

> **模型怎麼訓練：** 請用**同一支 iPhone、同樣的架設和打光**拍的影像在 Kaggle 上訓練。訓練時用 `--tile off`，因為相機實測是把檢測區整塊送進模型，不會切塊。

## 步驟 1：plan（速度還不知道也能先跑）

```bash
python camera_test/camtest.py plan -c config.yaml                        # 列出 20 ~ 1000 mm/s 各速度的結果
python camera_test/camtest.py plan -c config.yaml --speed 300            # 只看某個速度
python camera_test/camtest.py plan -c config.yaml --speeds 100,200,400   # 自訂要列的速度
```

輸出內容：

- **最高可用速度**，以及是哪個因素限制了它（快門太慢，或每件拍到的幀數不夠）。
- **每個速度的明細**：
  - 動態模糊幾像素
  - 每件完整拍到幾幀
  - 滾動快門歪斜多少
  - 需要多快的快門
  - 每分鐘幾件
- **缺陷大小檢查**：最小缺陷在影像上、以及縮到模型輸入後各有幾像素，不夠會告訴你要怎麼改。

## 步驟 2：架設 iPhone

| 項目 | 建議 |
|---|---|
| 固定 | 用支架鎖死，鏡頭和輸送帶平行。iPhone 不能用手拿 |
| 錄影 App | iPhone 內建相機**不能手動設快門**。請用能手動設定快門、ISO、對焦、白平衡的錄影 App，例如 Blackmagic Camera（免費） |
| 快門 | 照 `plan` 算出的「需要的快門」設定，通常 1/1000 ~ 1/4000 秒 |
| ISO / 光源 | 快門越快需要越多光，`plan` 會算出要比一般照明多幾倍。用 LED 補光燈，並把 ISO 壓低（雜訊少） |
| 對焦 / 白平衡 | 手動鎖定，不要自動 |
| 關掉 | 防手震（會裁切、扭曲畫面）、HDR、自動場景 |
| 解析度 / FPS | 和設定檔 `camera.resolution`、`camera.fps` 一致；橫式或直式也要一致 |
| 背景 | 素色輸送帶最好（`presence: background`）。有花紋時改用 `presence: model`，模型需要多一個「整個工件」的類別 |
| 散熱 / 電源 | 長時間錄影要接電源；手機過熱會自動降 FPS |

## 步驟 3：calib（放一把尺，量每像素幾 mm）

在工件的位置放一把尺，用 iPhone 拍一張照片（或錄一小段影片），然後：

```bash
python camera_test/camtest.py calib -c config.yaml --source ruler.jpg --distance-mm 100 --write
```

- 會跳出視窗，在尺上點相距 100 mm 的兩點，結果自動寫回設定檔的 `camera.mm_per_px`。
- 沒有視窗環境時，改用 `--points x1,y1,x2,y2` 直接給兩點座標。
- **換相機、換位置、換解析度都要重新 calib。** 每像素幾 mm 錯了，缺陷大小和速度就全錯。

## 步驟 4：錄影 + test（速度測試）

1. 準備一批**已知答案**的工件：良品和 NG 品都要有，並記住 NG 是第幾件經過。
2. 每個速度錄一段影片：
   - 開頭**至少 1 秒畫面是空的**（用來建立背景）。
   - 然後讓工件**依序**通過。
   - 結尾也留一點空畫面。
3. 填 `manifest.csv`（範例：`manifest.example.csv`）：

   ```csv
   video,speed_mm_s,n_parts,ng_parts
   run_slow.mov,100,20,"3,8,15"
   run_mid.mov,300,20,"3,8,15"
   run_fast.mov,,20,"3,8,15"
   ```

   | 欄位 | 填什麼 |
   |---|---|
   | `video` | 影片檔名 |
   | `speed_mm_s` | 產線速度；不知道就留空，程式會從影片量出來 |
   | `n_parts` | 這段影片總共經過幾件 |
   | `ng_parts` | NG 是第幾件，用逗號分隔 |

4. 執行：

   ```bash
   python camera_test/camtest.py test -c config.yaml --manifest manifest.csv
   ```

5. 看 `camtest_runs/test_日期時間/` 裡的輸出：

   | 檔案 | 內容 |
   |---|---|
   | `speed_test_report.md` | 每段影片的漏檢率、過殺率、需確認比例、量到的速度，以及**建議最高速度** |
   | `speed_vs_accuracy.png` | 速度 vs 漏檢率 / 過殺率曲線 |
   | `parts.csv` | 每個工件的判定、拍到幾幀、量到的速度 |
   | `parts/NG/`、`parts/CHECK/`、`parts/OK/` | 工件影像（良品是抽樣存）。人工確認後放進 Kaggle 資料集重訓，就是閉環 |

**建議最高速度怎麼決定：** 從最慢的速度開始往上看，第一個不合格的速度之前、最快的合格速度。

**合格標準：** 設定檔 `criteria` 決定，預設漏檢率 0%、過殺率 ≤ 2%。過殺率包含「需人工確認」，因為那也要花人力。

## 步驟 5：live（即時檢測）

```bash
python camera_test/camtest.py live -c config.yaml               # 用設定檔的 camera.source
python camera_test/camtest.py live -c config.yaml --source 1    # 換另一台攝影機
```

- **iPhone 當攝影機：** Mac 的「接續互通相機」可以把 iPhone 當成 Mac 的攝影機（需 macOS 13、iOS 16 以上），`source` 通常是 0 或 1。
  - **限制：** 這個模式**不能手動設快門**，所以 live 適合測流程或低速。
  - **高速：** 請用步驟 4 的錄影測試，或之後換成可手動設快門的相機（RTSP 監視鏡頭、工業相機）。
- **畫面顯示：**
  - 檢測區（藍框）
  - 偵測到的缺陷：紅色 = 超過門檻，黃色 = 接近門檻
  - 良品 / 不良 / 需確認的累計數
  - 處理速度 FPS
- **按鍵：** `q` 結束、`b` 重建背景（先把畫面清空）、`p` 暫停。
- 每個工件的判定會印在終端機。不良和需確認時會發出提示音，並存圖到 `camtest_runs/live_日期時間/`。

## 防呆機制

| 檢查 | 什麼情況會擋下來 |
|---|---|
| 設定檔 | 所有問題**一次列出**後停止，例如：缺欄位、數字超出合理範圍、快門比每幀時間還長、`min_valid_frames < vote_min_frames`、檢測區比工件短（工件永遠無法完整入鏡） |
| 單位填錯 | 速度填 0.5（看起來是 m/s）會提醒乘以 1000；快門填 2000 會自動當成 1/2000 秒並提醒 |
| 解析度 | 影片或相機的解析度和設定不同就停止（每像素 mm 會錯）；直式 / 橫式相反會特別指出 |
| 模型和門檻 | 門檻檔是分類模型的、門檻檔的類別不在模型裡、`presence: model` 但模型沒有工件類別 → 停止 |
| 背景 | 開頭畫面有東西在動、輸送帶有花紋、燈光閃爍 → 背景不穩定，停止並說明原因 |
| manifest | 找不到影片、件數不是正整數、NG 位置重複或超出範圍 → 全部列出 |
| 計數 | 數到的件數和答案不同 → 該段影片結果**作廢**（對不上就不能算漏檢率） |
| 宣告速度 | 和影片量到的差超過 25% → 提醒檢查 `mm_per_px` 或宣告速度 |
| 高速錯覺 | 每幀移動接近工件間距時，`plan` 直接判「不行」；工件看起來沒在動 → 判需人工確認 |
| 處理太慢 | live 時處理 FPS 低於相機 FPS，畫面上會出現紅字警告 |
| 相機斷線 | live 時自動重連；持續沒畫面就停止 |

## 判定規則（多幀投票）

| 情況 | 判定 |
|---|---|
| 完整拍到的幀數 < `min_valid_frames` | **需人工確認**（太快，看得不夠） |
| 長度超過設定的 1.6 倍 | **需人工確認**（疑似兩件黏在一起） |
| 影像太暗或過曝 | **需人工確認** |
| 看到缺陷的幀數 ≥ `vote_min_frames` | **不良（NG）** |
| 只有 1 ~ (`vote_min_frames` − 1) 幀看到缺陷 | **需人工確認**（不放行、也不直接判 NG） |
| 完全沒看到缺陷 | 良品（OK） |

## 換成監視鏡頭或工業相機

1. **改設定檔的 `camera` 段：**
   - `type`：`cctv` 或 `industrial`。工業相機是全域快門，沒有滾動快門歪斜。
   - `source`：例如 RTSP 網址或 USB 攝影機編號。
   - `resolution`、`fps`、`shutter` 照新相機設定。
2. **重新 `calib`，再跑一次 `plan` 和 `test`。** 新舊相機的速度測試報告可以直接比較。
3. **用新相機的影像重訓：** 不同相機的影像特性不同，建議在 Kaggle 上用新相機的影像重訓（閉環流程相同）。
4. **工業相機：** 目前支援 OpenCV 能開的來源（USB、RTSP、影片）。需要廠商 SDK（例如 Basler pypylon）的 GigE 相機，之後可以再加。

## 限制與已驗證範圍

- **單一產線：** 工件沿同一方向移動，前後工件之間在畫面上要有看得到的空隙。
- **滾動快門讀出時間：** iPhone 用估計值（20 ms），只影響 `plan` 的歪斜估算，不影響判定。
- **已驗證的範圍：**
  - 16 個單元測試，加上用合成輸送帶影片和小型 YOLO 模型做的完整測試（60 ~ 3000 mm/s）。
  - **還沒用真正的 iPhone 影片跑過。** 第一次實測時，請先用最慢的速度確認計數和判定正確。
