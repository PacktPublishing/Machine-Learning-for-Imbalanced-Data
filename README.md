# Machine Learning for Imbalanced Data

<a href="https://www.amazon.com/dp/1801070830"><img src="https://m.media-amazon.com/images/W/MEDIAX_792452-T2/images/I/41ldy7Kud6L._SX342_SY445_.jpg" alt="Machine Learning for Imbalanced Data" height="256px" align="right"></a>

This is the code repository for [Machine Learning for Imbalanced Data](https://www.amazon.com/dp/1801070830), published by Packt.

**Tackle imbalanced datasets using machine learning and deep learning techniques**

## What is this book about?

As machine learning practitioners, we often encounter imbalanced datasets in which one class has considerably fewer instances than the other. Many machine learning algorithms assume an equilibrium between majority and minority classes, leading to suboptimal performance on imbalanced data. This comprehensive guide helps you address this class imbalance to significantly improve model performance.

This book covers the following exciting features: 
* Use imbalanced data in your machine learning models effectively
* Explore the metrics used when classes are imbalanced
* Understand how and when to apply various sampling methods such as over-sampling and under-sampling
* Apply data-based, algorithm-based, and hybrid approaches to deal with class imbalance
* Combine and choose from various options for data balancing while avoiding common pitfalls
* Understand the concepts of model calibration and threshold adjustment in the context of dealing with imbalanced datasets

If you feel this book is for you, get your copy today!
* [Amazon.com link](https://www.amazon.com/Machine-Learning-Imbalanced-Data-imbalanced/dp/1801070830) 
* [Amazon.in link](https://www.amazon.in/Machine-Learning-Imbalanced-Data-imbalanced/dp/1801070830) 
* [Packt link](https://www.packtpub.com/product/machine-learning-for-imbalanced-data/9781801070836)

#### Download a free PDF
 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost. </i> <br>
 Please go to <a href="https://packt.link/free-ebook/9781801070836">this link</a> to claim your free PDF.

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
from collections import Counter
X, y = make_data(sep=2)
print(y.value_counts())
sns.scatterplot(data=X, x="feature_1", y="feature_2")
plt.title('Separation: {}'.format(separation))
plt.show()
```
**Following is what you need for this book:**
This book is for machine learning practitioners who want to effectively address the challenges of imbalanced datasets in their projects. Data scientists, machine learning engineers/scientists, research scientists/engineers, and data scientists/engineers will find this book helpful. Though complete beginners are welcome to read this book, some familiarity with core machine learning concepts will help readers maximize the benefits and insights gained from this comprehensive resource.

With the following software and hardware list you can run all code files present in the book (Chapter 1-10).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
|  	1-10	   |   	Google Colab                                  			  | Any OS | 		


### Questions or Feedback
If you have any questions or feedback, please feel free to use the [Discussions tab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/discussions) of this repository. You can start a new discussion under an appropriate category. 

### Related products <Other books you may enjoy>
* Machine Learning with PyTorch and Scikit-Learn  [[Packt]](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312) [[Amazon]](https://www.amazon.in/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319/ref=sr_1_3?keywords=Machine+Learning+with+PyTorch+and+Scikit-Learn&sr=8-3)
  
* Graph Machine Learning  [[Packt]](https://www.packtpub.com/product/graph-machine-learning/9781800204492) [[Amazon]](https://www.amazon.in/Graph-Machine-Learning-techniques-algorithms/dp/1800204493/ref=sr_1_3?keywords=Graph+Machine+Learning&sr=8-3)
  
## Get to Know the Authors
**Kumar Abhishek** is a seasoned Senior Machine Learning Engineer, specializing in risk analysis and fraud detection. With over a decade of experience at companies such as Expedia, Microsoft, Amazon, and a Bay Area startup, Kumar holds an MS in Computer Science from the University of Florida.

**Dr. Mounir Abdelaziz** is a deep learning researcher specializing in computer vision applications. He holds a Ph.D. in computer science and technology from Central South University, China. During his Ph.D. journey, he developed innovative algorithms to address practical computer vision challenges. He has also authored numerous research articles in the field of few-shot learning for image classification.

## Table of Contents and Code Notebooks

1. Introduction to Data Imbalance in Machine Learning [[open dir](chapter01)] 
2. Oversampling Methods [[open dir](chapter02)] 
3. Undersampling Methods [[open dir](chapter03)] 
4. Ensemble Methods [[open dir](chapter04)] 
5. Cost-Sensitive Learning [[open dir](chapter05)] 
6. Data Imbalance in Deep Learning [[open dir](chapter06)]
7. Data-Level Deep Learning Methods [[open dir](chapter07)] 
8. Algorithm-Level Deep Learning Techniques [[open dir](chapter08)]  
9. Hybrid Deep Learning Methods [[open dir](chapter09)] 
10. Model Calibration [[open dir](chapter10)]

[Detailed Table of content](Table&#32;of&#32;Content.pdf)


## List of notebooks

| Notebook ID | Description | Link |
|-------------|-------------|------|
| Notebook 1.1 | Imbalanced-learn demo | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/tree/main/chapter01/imblearn_demo.ipynb) |
| Notebook 2.1 | Oversampling techniques | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter02/chapter02.ipynb) |
| Notebook 2.2 | Oversampling performance | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter02/model-perf-comparison.ipynb) |
| Notebook 2.3 | SMOTE problems | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter02/smote-plots.ipynb) |
| Notebook 3.1 | Various undersampling techniques | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter03/chapter03.ipynb) |
| Notebook 3.2 | Undersampling performance | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter03/model_perf_comparison_undersampling.ipynb) |
| Notebook 4.1 | Ensemble techniques overview | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter04/chapter04.ipynb) |
| Notebook 4.2 | Ensembling methods performance | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter04/model_perf_comparison_ensembling.ipynb) |
| Notebook 5.1 | Class weight with Sklearn/XGBoost | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter05/chapter05.ipynb) |
| Notebook 5.2 | Threshold tuning techniques | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter05/threshold_computation.ipynb) |
| Notebook 6.1 | Simple neural network | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter06/chapter06.ipynb) |
| Notebook 6.2 | Multi-class classification | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter06/multiclass_classification_PR_curve.ipynb) |
| Notebook 7.1 | Augmix on FashionMNIST | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/Augmix_FashionMNIST.ipynb) |
| Notebook 7.2 | Cutmix, Mixup, Remix on FashionMNIST | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/Cutmix_mixup_remix_FashionMNIST.ipynb) |
| Notebook 7.3 | NLP data-level techniques | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/Data_level_techniques_NLP.ipynb) |
| Notebook 7.4 | Dynamic sampling | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/Dynamic_sampler.ipynb) |
| Notebook 7.5 | VAE with MNIST | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/VAE_MNIST.ipynb) |
| Notebook 7.6 | Cutmix technique | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/cutmix.ipynb) |
| Notebook 7.7 | Cutout technique | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/cutout.ipynb) |
| Notebook 7.8 | Mixup technique | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/mixup.ipynb) |
| Notebook 7.9 | Data transformation plotting | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter07/plot_transforms.ipynb) |
| Notebook 8.1 | CIFAR10 focal loss | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter08/CIFAR10_LT_Focal_Loss.ipynb) |
| Notebook 8.2 | CDT loss implementation | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter08/Class_Dependent_Temperature_(CDT)_Loss.ipynb) |
| Notebook 8.3 | Class balanced loss | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter08/Class_balanced_loss.ipynb) |
| Notebook 8.4 | Class-wise difficulty balanced loss | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter08/Class_wise_difficulty_balanced_loss.ipynb) |
| Notebook 8.5 | DRW technique | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter08/Deferred_reweighting_DRW.ipynb) |
| Notebook 8.6 | Tweet emotion detection | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter08/Tweet_emotion_detection_using_class_weights_Huggingface.ipynb) |
| Notebook 8.7 | PyTorch class weighting | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter08/class_weighting_pytorch_imbalanced_dataset.ipynb) |
| Notebook 9.1 | GNN demo | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter09/GNNs.ipynb) |
| Notebook 9.2 | OHEM technique | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter09/Online_hard_example_mining-OHEM.ipynb) |
| Notebook 9.3 | Class rectification loss | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter09/Class_Rectification_Loss.ipynb) |
| Notebook 10.1 | Calibration techniques | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter10/calibration_MNIST.ipynb) |
| Notebook 10.2 | Sampling/weighting impact on calibration | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter10/calibration_to_account_for_sampling_or_weighting.ipynb) |
| Notebook 10.3 | Imbalance handling impact on calibration | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter10/model-calibration.ipynb) |
| Notebook 10.4 | Kaggle HR data calibration | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter10/model_calibration_on_Kaggle_HR_data.ipynb) |
| Notebook 10.5 | Plat's scaling and isotonic regression | [ipynb/colab](https://github.com/PacktPublishing/Machine-Learning-for-Imbalanced-Data/blob/main/chapter10/platts_scaling_and_isotonic_regression.ipynb) |

---

<br>
<br>

Kumar Abhishek, Dr. Mounir Abdelaziz, *Machine Learning for Imbalanced Data*. Packt Publishing, 2023.

        @book{mlimbdata2023,
        title = {Machine Learning for Imbalanced Data},
        author = {Kumar Abhishek and Mounir Abdelaziz},
        year = {2023},
        publisher = {Packt},
        isbn = {9781801070836}
    }
