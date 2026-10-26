# visual-inspection
Visual inspection using smartphone and CCTV cameras - 手機/監視鏡頭視覺檢測

## Kaggle 訓練程式（YOLO26）

[`kaggle/`](kaggle/README.md)：在 Kaggle 免費 GPU 上訓練與評估 YOLO 檢測模型。

- `kaggle/medical_yolo.py`：醫學影像（DICOM、CSV 框標註、分割遮罩、分類）
- `kaggle/semiconductor_yolo.py`：工業 / 半導體（大圖小缺陷切塊、VOC / COCO、WM-811K 晶圓圖）
- 內建分組切分（避免資料洩漏）、逐類調門檻、漏檢率 / 過殺率報告、新舊模型上線關卡、主動學習複判佇列

## 相機實測工具（iPhone → 監視鏡頭 / 工業相機）

[`camera_test/`](camera_test/README.md)：用 Kaggle 訓練好的模型實測相機。

- `plan`：依產線速度、快門、FPS、視野算出能不能檢、最高可用速度
- `calib`：放尺量每像素幾 mm
- `test`：錄影速度測試，和答案比對，得到「速度 vs 漏檢率 / 過殺率」與建議最高速度
- `live`：接相機即時檢測（Mac 上可用 iPhone 當攝影機）
- 全程防呆：設定錯誤一次列出；拿不準的工件判「需人工確認」，不會放行
