



> [Wang Y., Huang G., Song S., Pan X., Xia Y. and Wu C.  Regularizing Deep Networks with Semantic Data Augmentation. TPAMI.](https://arxiv.org/pdf/2007.10538.pdf)


## Usage

```
python train.py resnet32 cifar10 --leverage=0.1
```

Note: I tried "--leverage=0.5" following the official but nan error happened (as covariance matrix diverges).