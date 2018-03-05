# **Exposure: A White-Box Photo Post-Processing Framework [[Paper](https://arxiv.org/abs/1709.09602)]**
[Yuanming Hu](http://taichi.graphics/me/), Microsoft Research & MIT CSAIL

[Hao He](https://github.com/hehaodele), Microsoft Research & MIT CSAIL

Chenxi Xu, Microsoft Research & Peking University

[Baoyuan Wang](https://sites.google.com/site/zjuwby/), Microsoft Research

[Stephen Lin](https://www.microsoft.com/en-us/research/people/stevelin/),  Microsoft Research

**(Page under Construction. Code coming soon! Last update: March 5, 2018)**

ACM Transactions on Graphics (to be presented at SIGGRAPH 2018)

<img src="web/images/teaser.jpg">

# Install
Requirements: `python3` and `tensorflow`
```
git clone https://github.com/yuanming-hu/exposure
pip3 install tensorflow-gpu tifffile sklearn scikit-image exifread
```


Make sure you have `pdflatex`, if you want to generate the steps.

# Use the pretrained model
 - `python3 evaluate.py example a.jpg b.png c.tiff`

# Train your own model!
(More detailed instructions coming.)
  - Download and setup the [`MIT-Adobe FiveK Dataset`](https://data.csail.mit.edu/graphics/fivek/)
  - `python3 train.py example` (This will load config_example.py)
  - Have a cup of tea (2 hours on a GTX 1080 Ti) 
  - `python3 evaluate.py example a.jpg b.png c.tiff`

# Visual Results

<img src="web/images/fig02.jpeg" width="400"> <img src="web/images/fig04.jpeg" width="400">
<img src="web/images/fig09.jpeg" width="800">
<img src="web/images/fig11.jpeg" width="800">
<img src="web/images/fig13.jpeg" width="474">
<img src="web/images/fig17.png" width="326">
