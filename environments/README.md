# environments/ — conda 環境定義

FoveaMIL は GPU のアーキテクチャ世代によって必要な CUDA / PyTorch が変わるため，**CUDA 版ごとに
2 つの環境**を用意する．環境名・ファイル名は CUDA 版で表す．

## どちらを使うか（GPU → 環境）

| GPU | アーキ | compute capability | 使う環境 |
|---|---|---|---|
| GTX 1080Ti | Pascal | sm_61 | **cu118** |
| RTX A6000 | Ampere | sm_86 | **cu118**（推奨）/ cu128 どちらも可 |
| RTX 5070Ti | Blackwell | sm_120 | **cu128** |

理由: **Blackwell (sm_120) は CUDA 12.8+ が必須**，**Pascal (sm_61) は CUDA 12.8 ビルドから外れる**ため，
1 つの PyTorch ビルドで両端を賄えない．Ampere (sm_86) は両方の arch に含まれるのでどちらでも動く．

| 環境 | python | torch / CUDA | arch（含む sm） |
|---|---|---|---|
| `foveamil-cu118` | 3.8 | torch 2.4.x / cu118 | 50,60,**61**,70,75,80,**86**,90 |
| `foveamil-cu128` | 3.11 | torch 2.9.x / cu128 | 70,75,80,**86**,90,100,**120** |

## 作成

```bash
# Pascal / Ampere GPU
conda env create -f environments/environment-cu118.yml
conda activate foveamil-cu118
pip install -e .                      # foveamil パッケージを editable 導入

# Blackwell GPU
conda env create -f environments/environment-cu128.yml
conda activate foveamil-cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```
