# yolo_cnn_pipeline_v2 —— YOLO + CNN 两阶段 pipeline + 视频可视化

独立文件夹，**不依赖** `cnn_synthetic_v2/` 或原仓库。继承自上一阶段的 4 个 CNN 分类器（用 AIMv2 + DINOv3），在 V2 数据上再训 YOLO（v8n + 11n），做两阶段 pipeline：

```
视频帧 / 图片  →  YOLO 找框  →  对每个框裁 crop  →  CNN 分类  →  标注输出
```

## 5 个脚本一句话

| 脚本 | 干啥 |
|---|---|
| `make_yolo_split.py` | 把 V2 拆成 train/val/test YOLO 目录（与 CNN 同 split，seed=42）|
| `train_yolo.py` | 训 YOLOv8n 或 YOLO11n，输出 `runs/<name>/weights/best.pt` |
| `eval_pipeline.py` | **V2 内部** test set 上算"YOLO mAP / YOLO 分类 / CNN 分类"三个指标 |
| `cross_domain_eval.py` | **真实 Roboflow** 数据上跑跨域评估（CNN-only + YOLO-only + Pipeline）|
| `infer_video.py` | 视频逐帧 YOLO+CNN 标注，输出 mp4 |

## 上云完整跑法

### 0. 准备本地文件

```
本地需要这些上传到云：
  yolo_cnn_pipeline_v2.tar.gz        # 本文件夹打包
  synthetic_v2.tar.gz                # V2 数据（之前用过的）
  cnn_aimv2_best.pt                  # AIMv2 best checkpoint (3.7 GB)
  cnn_dinov3_best.pt                 # DINOv3 best checkpoint (1.0 GB)
  Final-Test.mp4                     # Joe 的测试视频（先从 Dropbox 下）
```

CNN best.pt 在 `cnn_synthetic_v2/runs/cls_syn_v2_aimv2/best.pt` 和 `.../cls_syn_v2_dinov3/best.pt`。

### 1. 本地打包 + 上传

```powershell
# Windows PowerShell
cd "E:/课程/cs7643/group_work/Pokemon-Visualization-Tool-main/Pokemon-Visualization-Tool-main"
tar --exclude='yolo_cnn_pipeline_v2/runs' --exclude='yolo_cnn_pipeline_v2/runs_eval' --exclude='yolo_cnn_pipeline_v2/videos' --exclude='yolo_cnn_pipeline_v2/yolo_data' -czf yolo_cnn_pipeline_v2.tar.gz yolo_cnn_pipeline_v2

# 复制 CNN 权重为简单文件名（避免上云时路径太长）
copy cnn_synthetic_v2\runs\cls_syn_v2_aimv2\best.pt cnn_aimv2_best.pt
copy cnn_synthetic_v2\runs\cls_syn_v2_dinov3\best.pt cnn_dinov3_best.pt

# 上传
scp -P <port> yolo_cnn_pipeline_v2.tar.gz root@<ip>:~/
scp -P <port> synthetic_v2.tar.gz root@<ip>:~/        # 只在云上没有时传
scp -P <port> cnn_aimv2_best.pt root@<ip>:~/
scp -P <port> cnn_dinov3_best.pt root@<ip>:~/
scp -P <port> Final-Test.mp4 root@<ip>:~/
```

### 2. 云上一键脚本

ssh 进云之后，整段贴：

```bash
cat > ~/run_pipeline.sh << 'EOF'
#!/bin/bash
set -e
echo "========== $(date) START =========="

cd ~
echo "[1/5] 解压代码 + 数据"
tar -xzf yolo_cnn_pipeline_v2.tar.gz
if [ ! -d "pokemon_data/Synthetic V2 (1)" ]; then
    mkdir -p pokemon_data
    tar -xzf synthetic_v2.tar.gz -C pokemon_data/
fi

echo "[2/5] 装依赖"
cd ~/yolo_cnn_pipeline_v2
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  -> 已有 CUDA torch，只装附加依赖"
    pip install "ultralytics>=8.3" "timm>=1.0.11" Pillow PyYAML numpy tqdm matplotlib scikit-learn opencv-python
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install "ultralytics>=8.3" "timm>=1.0.11" Pillow PyYAML numpy tqdm matplotlib scikit-learn opencv-python
fi
python -c "import torch, ultralytics, timm, cv2; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'ultralytics:', ultralytics.__version__, 'timm:', timm.__version__, 'cv2:', cv2.__version__)"

echo "[3/5] 切分 YOLO 数据"
python make_yolo_split.py --source "/root/pokemon_data/Synthetic V2 (1)/Synthetic" --output yolo_data

echo "[4/5] 训练 2 个 YOLO（v8n + 11n）"
python train_yolo.py --config configs/yolov8n.yaml --data yolo_data/data.yaml
python train_yolo.py --config configs/yolo11n.yaml --data yolo_data/data.yaml

echo "[5/5] Pipeline eval + 视频"
mkdir -p runs_eval videos
for yolo in yolov8n_v2 yolo11n_v2; do
  for cnn_name in aimv2 dinov3; do
    echo "==== eval ${yolo} + ${cnn_name} ===="
    python eval_pipeline.py \
      --yolo runs/${yolo}/weights/best.pt \
      --cnn ~/cnn_${cnn_name}_best.pt \
      --data yolo_data/data.yaml --split test \
      --out runs_eval/${yolo}_${cnn_name}

    echo "==== video ${yolo} + ${cnn_name} ===="
    python infer_video.py \
      --yolo runs/${yolo}/weights/best.pt \
      --cnn ~/cnn_${cnn_name}_best.pt \
      --video ~/Final-Test.mp4 \
      --out videos/${yolo}_${cnn_name}.mp4
  done
done

echo "[final] 打包结果"
cd ~/yolo_cnn_pipeline_v2
tar -czf ~/pipeline_results.tar.gz --exclude='*/weights/*.pt' runs/ runs_eval/ videos/
tar -czf ~/pipeline_yolo_ckpt.tar.gz runs/*/weights/best.pt
ls -lh ~/pipeline_results.tar.gz ~/pipeline_yolo_ckpt.tar.gz

echo "========== $(date) ALL DONE =========="
touch ~/PIPELINE_DONE.txt
EOF
chmod +x ~/run_pipeline.sh
```

### 3. tmux 启动

```bash
tmux new -d -s pipeline "bash ~/run_pipeline.sh 2>&1 | tee ~/run_pipeline.log"
tmux ls && sleep 5 && tail -30 ~/run_pipeline.log
exit
```

预估 3-5 小时（4090）。

### 4. 完成后回来

```bash
ssh -p <port> root@<ip>
ls ~/PIPELINE_DONE.txt && echo "OK"
ls -lh ~/pipeline_results.tar.gz ~/pipeline_yolo_ckpt.tar.gz
exit
```

```powershell
# 本地拉
scp -P <port> root@<ip>:~/pipeline_results.tar.gz .
scp -P <port> root@<ip>:~/pipeline_yolo_ckpt.tar.gz .

tar -xzf pipeline_results.tar.gz -C yolo_cnn_pipeline_v2/
tar -xzf pipeline_yolo_ckpt.tar.gz -C yolo_cnn_pipeline_v2/

# 验证
ls yolo_cnn_pipeline_v2/runs/*/weights/best.pt
ls yolo_cnn_pipeline_v2/runs_eval/*/pipeline_metrics.json
ls yolo_cnn_pipeline_v2/videos/*.mp4
```

## 输出位置

```
yolo_cnn_pipeline_v2/
├── runs/                        ← YOLO 训练产物
│   ├── yolov8n_v2/
│   │   ├── weights/best.pt
│   │   ├── results.csv
│   │   ├── confusion_matrix.png
│   │   ├── test_metrics.json    ← 我们的 wrapper 写的
│   │   └── (Ultralytics 自带的一堆图)
│   └── yolo11n_v2/
│       └── (同上)
├── runs_eval/                   ★ pipeline 主结果
│   ├── yolov8n_v2_aimv2/
│   │   ├── pipeline_metrics.json    ← 论文主表数字
│   │   ├── per_box_predictions.csv
│   │   ├── confusion_matrix_yolo.png
│   │   └── confusion_matrix_cnn.png
│   ├── yolov8n_v2_dinov3/
│   ├── yolo11n_v2_aimv2/
│   └── yolo11n_v2_dinov3/
└── videos/                      ★ 视频可视化
    ├── yolov8n_v2_aimv2.mp4
    ├── yolov8n_v2_dinov3.mp4
    ├── yolo11n_v2_aimv2.mp4
    └── yolo11n_v2_dinov3.mp4
```

## 论文主表（4 行）

| YOLO | CNN | YOLO mAP50 | YOLO cls (TP) | CNN cls (TP) | Agreement |
|---|---|---|---|---|---|
| YOLOv8n | AIMv2 | ... | ... | ... | ... |
| YOLOv8n | DINOv3 | ... | ... | ... | ... |
| YOLO11n | AIMv2 | ... | ... | ... | ... |
| YOLO11n | DINOv3 | ... | ... | ... | ... |

数字都在 `runs_eval/*/pipeline_metrics.json` 里。

## 常见问题

### Q1: HuggingFace 下载预训练 YOLOv8/11 权重失败
```bash
# 在脚本第一步加：
export HF_ENDPOINT=https://hf-mirror.com
```
或者本地下载 `yolov8n.pt` / `yolo11n.pt` 后传上云，把 config 里 `model_pt` 改成绝对路径。

### Q2: 显存不够
改 yaml 里 `batch: 16` → `batch: 8` 或 `4`。

### Q3: 视频很大跑很慢
用 `--every-n 3` 之类（每 3 帧推理一次，中间帧用上一次结果）—— 速度 3x，视觉无明显降级。
或者 `--max-frames 1000` 只推理前 1000 帧。

### Q4: timm 模型名不对
进 python 跑 `timm.list_models('*aimv2*')` 看现版本支持的名字。当前 plan 用：
- AIMv2: `aimv2_large_patch14_224.apple_pt`
- DINOv3: `vit_base_patch16_dinov3.lvd1689m`
（CNN 的 ckpt 里写死了名字，**云上 timm 版本要 ≥ 1.0.11**）
