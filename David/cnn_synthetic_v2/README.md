# cnn_synthetic_v2 —— Pokemon Snap 分类基线（V2 合成数据）

独立文件夹，不依赖原仓库。用 V2 合成数据（sprite 贴 Pokemon Snap 背景，YOLO 格式）按 bbox 裁 crop 做**单标签分类**，对比 4 个模型：

| 模型 | 角色 |
|---|---|
| ResNet50 | baseline |
| DINOv3 ViT-B/14 | SOTA 自监督（跨域鲁棒性） |
| AIMv2-Large | SOTA 自回归预训练 |
| SigLIP 2-Base | SOTA 对比预训练 |

## 上云跑法

### 1. 上传

在本地把本文件夹打包或 scp 上云。V2 数据单独传（zip 或解压好的目录都行）。

```bash
# 本地
scp -r cnn_synthetic_v2 root@<ip>:~/
# V2 数据（zip 或已解压目录）任选
scp "Synthetic V2 (1).zip" root@<ip>:~/pokemon_data/
```

### 2. 在云上装依赖

```bash
cd ~/cnn_synthetic_v2
pip install -r requirements.txt
```

> 如果云服务器里已有 PyTorch 环境，可以只装差异：`pip install timm>=1.0.11 scikit-learn tqdm pyyaml matplotlib Pillow`

### 3. 解压 V2 数据（如果还没解压）

```bash
cd ~/pokemon_data
unzip -q "Synthetic V2 (1).zip"
# 解压后的结构应该是：
#   Synthetic V2 (1)/Synthetic/images/*.jpg
#   Synthetic V2 (1)/Synthetic/labels/*.txt
#   Synthetic V2 (1)/Synthetic/data.yaml
```

### 4. 转数据（一次性）

```bash
cd ~/cnn_synthetic_v2
python convert.py --source "/root/pokemon_data/Synthetic V2 (1)/Synthetic" --output-dir ./data
```

跑完会生成：
- `data/manifest.csv` —— 每行一个 crop（`image_path, label, label_id, split, source_image`）
- `data/label_to_id.json` —— 12 类名到 id
- `data/summary.json` —— 统计信息
- `data/crops/<label>/*.jpg` —— 裁好的 crop

默认参数（如需调整）：
- `--padding 0.1` —— bbox 外扩 10% 带点上下文
- `--min-bbox 16` —— 丢掉 w 或 h < 16 像素的过小 crop
- `--train-ratio/val-ratio/test-ratio` —— 默认 0.8/0.1/0.1，**按源图 id 切分**避免泄漏

### 5. 训练 4 个模型（串行）

```bash
python train.py --config configs/resnet50.yaml
python train.py --config configs/dinov3.yaml
python train.py --config configs/aimv2.yaml
python train.py --config configs/siglip2.yaml
```

4090 上预期每个 20-60 分钟（AIMv2-Large 最慢）。**第一次跑各 timm 模型会从 HuggingFace 下载预训练权重**，占 400MB-1.5GB 每个，耐心等。

### 6. 看结果

每个实验在 `runs/<experiment_name>/` 下：

```
runs/cls_syn_v2_resnet50/
├── best.pt                   # 最佳 checkpoint（下一轮跨域测试要用，留着）
├── last.pt                   # 最后一个 checkpoint
├── history.json              # 每 epoch 的 train/val loss/acc/f1
├── test_metrics.json         # 测试集 accuracy / macro_f1 / per-class / confusion_matrix
├── test_predictions.csv      # 每个测试样本的预测（方便分析错分）
├── confusion_matrix.png      # 可视化混淆矩阵
├── config.snapshot.yaml      # 实验用的 config 快照
└── train.log                 # 完整训练日志
```

## 调参建议

### 显存不够 (OOM)？

改对应 config 里的 `batch_size`：
- `aimv2.yaml` 从 32 → 16
- `dinov3.yaml` / `siglip2.yaml` 从 64 → 32

### 训练时间太长？

先跑 ResNet50 + DINOv3 两个（最有代表性对比），AIMv2/SigLIP 2 看剩多少时间。

### 想确认超参没错？

每个 config 临时改 `epochs: 1` 跑一轮 smoke test，确认流程通 + 各输出文件都生成，再调回 15。

## 下一轮（不在本包范围）

- 用 `best.pt` 在**真实游戏画面**上做 crop 分类测试（验证 domain gap）
- 需要真实画面的 bbox 标注（等 Joe 的 Roboflow 标注，或自己在 Pokemon Snap 视频里挑几十帧手标）
- 新写一个 `eval_real.py` 脚本即可，同样独立，不影响本包

## 常见问题

**Q: timm 模型名找不到？**
A: timm 版本更新后模型名可能变。进 python 交互：
```python
import timm
print(timm.list_models('*dinov3*'))
print(timm.list_models('*aimv2*'))
print(timm.list_models('*siglip*'))
```
把 config 里的 `model.name` 换成列出的可用名。

**Q: 数据路径报错？**
A: `convert.py --source` 要指向包含 `images/ labels/ data.yaml` 的那个目录（不是 zip 解压后最外层）。核对：`ls <source>/data.yaml` 能看到。

**Q: HuggingFace 下载预训练权重失败？**
A: 云机可能屏蔽 HF，试着设镜像：`export HF_ENDPOINT=https://hf-mirror.com`，然后重跑。
