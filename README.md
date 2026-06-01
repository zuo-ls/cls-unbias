# Cls-unbias: class-unbiased model for medical diagnosis

The official PyTorch implementation of Cls-unbias, "Class Unbiasing for Generalization in Medical Diagnosis" [link1](https://arxiv.org/pdf/2508.06943); [link2](https://www.sciencedirect.com/science/article/abs/pii/S0031320326009337).


## What is Cls-unbias?
Medical diagnosis might fail due to bias. In this work, we identified **class-feature bias**, which refers to models' potential reliance on features that are strongly correlated with only a subset of classes, leading to biased performance and poor generalization on other classes. We aim to train a class-unbiased model (Cls-unbias) that mitigates both class imbalance and class-feature bias simultaneously.

## What's in this Repo?
This experiment gives an example of class-shared and class-specific features, demonstrates both the existence of the class-feature bias and the effectiveness of our method in addressing it.

This demo runs the toy experiments that quickly validates the paper's conclusions on a single CPU. The code is adaptable to other datasets.

## How to run the experiment?
Simply run
```bash
python main.py
```
will be able to get all results of the toy experiments presented in the paper. 

## Dependencies
Install dependencies:

```bash
pip install -r requirements.txt
```

## Cite Our Paper
```
@article{cls_unbias,
  title={Class unbiasing for generalization in medical diagnosis},
  author={Zuo, Lishi and Yi, Lu and Tu, Youzhi and Mak, Man-Wai},
  journal={Pattern Recognition},
  pages={113968},
  year={2026},
  publisher={Elsevier}
}
```
