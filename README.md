# Semi-supervised-training
This repo implements the semi-supervised training for medical image analysis. This code generally follows [Mean Teacher Pytorch implementation](https://github.com/CuriousAI/mean-teacher/tree/master/pytorch).

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch](https://pytorch.org/) (Recommended version 9.2)
- [Python 3](https://www.python.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## Dataset
We download dataset from [imageclef]()

## Run experiments
Please see example in the recipes for the code.

## Result on ImageCLEF2016
|     Percentage of labels (%)     | Accuracy ↓ (%) |
|:--------------:|:----------------:|
| 10% |  76.3    |
| 25% | 79.8  |
| 50% |  79.2 |
| 75%  |  82.1 |



