---
license: apache-2.0
datasets:
- Junwei-Xi/DDA-Training-Set
---
# Dual Data Alignment (NeurIPS'25 Spotlight)

This repository contains the official checkpoint (`DDA_ckpt.pth`) for the paper **"Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable"**, accepted by **NeurIPS 2025 as a Spotlight**.

[![GitHub](https://img.shields.io/badge/GitHub-Project%20Page-black?logo=github)](https://github.com/roy-ch/Dual-Data-Alignment)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14359-b31b1b.svg)](https://arxiv.org/abs/2505.14359)

## 📄 Model Details

- **Model File**: `DDA_ckpt.pth`
- **Paper**: [Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](https://arxiv.org/abs/2505.14359)

## 🚀 Performance

DDA achieves state-of-the-art performance across 11 benchmarks, including 4 in-the-wild datasets.

| Benchmark | NPR (CVPR'24) | UnivFD (CVPR'23) | FatFormer (CVPR'24) | SAFE (KDD'25) | C2P-CLIP (AAAI'25) | AIDE (ICLR'25) | DRCT (ICML'24) | AlignedForensics (ICLR'25) | DDA (ours) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GenImage (1G + 7D) | 51.5 ± 6.3 | 64.1 ± 10.8 | 62.8 ± 10.4 | 50.3 ± 1.2 | 74.4 ± 8.4 | 61.2 ± 11.9 | 84.7 ± 2.7 | 79.0 ± 22.7 | **91.7 ± 7.8** |
| DRCT-2M (16D) | 37.3 ± 15.0 | 61.8 ± 8.9 | 52.2 ± 5.7 | 59.3 ± 19.2 | 59.2 ± 9.9 | 64.6 ± 11.8 | 90.5 ± 7.4 | 95.5 ± 6.1 | **98.1 ± 1.4** |
| DDA-COCO (5D) | 42.2 ± 5.4 | 52.4 ± 1.5 | 51.7 ± 1.5 | 49.9 ± 0.3 | 51.3 ± 0.6 | 50.0 ± 0.4 | 60.2 ± 4.3 | 86.5 ± 19.1 | **92.2 ± 10.6** |
| EvalGEN (3D + 2AR) | 2.9 ± 2.7 | 15.4 ± 14.2 | 45.6 ± 33.1 | 1.1 ± 0.6 | 38.9 ± 31.2 | 19.1 ± 11.1 | 77.8 ± 5.4 | 68.0 ± 20.7 | **97.2 ± 4.2** |
| Synthbuster (9D) | 50.0 ± 2.6 | 67.8 ± 14.4 | 56.1 ± 10.7 | 46.5 ± 20.8 | 68.5 ± 11.4 | 53.9 ± 18.6 | 84.8 ± 3.6 | 77.4 ± 25.0 | **90.1 ± 5.6** |
| ForenSynths (11G) | 47.9 ± 22.6 | 77.7 ± 16.1 | 90.0 ± 11.8 | 49.7 ± 2.7 | **92.0 ± 10.1** | 59.4 ± 24.6 | 73.9 ± 13.4 | 53.9 ± 7.1 | 81.4 ± 13.9 |
| AIGCDetectionBenchmark (7G + 10D) | 53.1 ± 12.2 | 72.5 ± 17.3 | 85.0 ± 14.9 | 50.3 ± 1.1 | 81.4 ± 15.6 | 63.6 ± 13.9 | 81.4 ± 12.2 | 66.6 ± 21.6 | **87.8 ± 12.6** |
| Chameleon (Unknown) | 59.9 | 50.7 | 51.2 | 59.2 | 51.1 | 63.1 | 56.6 | 71.0 | **82.4** |
| Synthwildx (3D) | 49.8 ± 10.0 | 52.3 ± 11.3 | 52.1 ± 8.2 | 49.1 ± 0.7 | 57.1 ± 4.2 | 48.8 ± 0.8 | 55.1 ± 1.8 | 78.8 ± 17.8 | **90.9 ± 3.1** |
| WildRF (Unknown) | 63.5 ± 13.6 | 55.3 ± 5.7 | 58.9 ± 8.0 | 57.2 ± 18.5 | 59.6 ± 7.7 | 58.4 ± 12.9 | 50.6 ± 3.5 | 80.1 ± 10.3 | **90.3 ± 3.5** |
| Bfree-Online (Unknown) | 49.5 | 49.0 | 50.0 | 50.5 | 50.0 | 53.1 | 55.7 | 68.5 | **95.1** |
| **Avg ACC** | 46.1 ± 16.1 | 56.3 ± 16.5 | 59.6 ± 14.6 | 47.6 ± 16.0 | 62.1 ± 15.6 | 54.1 ± 12.8 | 70.1 ± 14.6 | 75.0 ± 11.1 | **90.7 ± 5.3** |
| **Min ACC** | 2.9 | 15.4 | 45.6 | 1.1 | 38.9 | 19.1 | 50.6 | 53.9 | **81.4** |

*Notably, DDA is the first detector to achieve over 80% cross-data accuracy on the Chameleon benchmark.*

## 📚 Citation

If you find this model useful, please cite our paper:
```code
@inproceedings{chen2025dual,
  title={Dual Data Alignment Makes {AI}-Generated Image Detector Easier Generalizable},
  author={Ruoxin Chen and Junwei Xi and Zhiyuan Yan and Ke-Yue Zhang and Shuang Wu and Jingyi Xie and Xu Chen and Lei Xu and Isabel Guan and Taiping Yao and Shouhong Ding},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={[https://openreview.net/forum?id=C39ShJwtD5](https://openreview.net/forum?id=C39ShJwtD5)}
}
```