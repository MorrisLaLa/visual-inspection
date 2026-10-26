# visual-inspection
Visual inspection using smartphone and CCTV cameras - 手機/監視鏡頭視覺檢測

## Kaggle 訓練程式（YOLO26）

[`kaggle/`](kaggle/README.md)：在 Kaggle 免費 GPU 上訓練與評估 YOLO 檢測模型。

- `kaggle/medical_yolo.py`：醫學影像（DICOM、CSV 框標註、分割遮罩、分類）
- `kaggle/semiconductor_yolo.py`：工業 / 半導體（大圖小缺陷切塊、VOC / COCO、WM-811K 晶圓圖）
- 內建分組切分（避免資料洩漏）、逐類調門檻、漏檢率 / 過殺率報告、新舊模型上線關卡、主動學習複判佇列
