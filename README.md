"This repo is about anomaly detection on data consisting of images. The implementation is generalizable to any image dataset. The anomaly detector is based on BiGAN architecture, and the anomaly score is computed according to the following paper:
```
@article{DBLP:journals/corr/abs-1802-06222,
  author       = {Houssam Zenati and
                  Chuan Sheng Foo and
                  Bruno Lecouat and
                  Gaurav Manek and
                  Vijay Ramaseshan Chandrasekhar},
  title        = {Efficient GAN-Based Anomaly Detection},
  journal      = {CoRR},
  volume       = {abs/1802.06222},
  year         = {2018},
  url          = {http://arxiv.org/abs/1802.06222},
  eprinttype    = {arXiv},
  eprint       = {1802.06222},
  timestamp    = {Mon, 13 Aug 2018 16:46:17 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1802-06222.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Two datasets were considered:

- the training dataset : an Inlier dataset that contains only one category of digits (e.g., 0)
the test dataset : an Outlier dataset that randomly samples MNIST images of other categories with a proportion of 20% taken as outliers.

The training aims to indirectly learn the probability density of handwritten digit 0 given some latent variable lying in $z_dim$-dimensional space.

An anomaly detector maps each data point to an anomaly score defined in the aforementioned paper."

