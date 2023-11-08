# Machine Learning for Imbalanced Data

<a href="https://www.packtpub.com/product/machine-learning-for-imbalanced-data/9781801070836"><img src="https://m.media-amazon.com/images/I/71OBftlbuDL._SL1500_.jpg" alt="Machine Learning for Imbalanced Data" height="256px" align="right"></a>

This is the code repository for [Machine Learning for Imbalanced Data](https://www.packtpub.com/product/machine-learning-for-imbalanced-data/9781801070836), published by Packt.

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

If you feel this book is for you, get your [copy](https://www.amazon.in/Machine-Learning-Imbalanced-Data-imbalanced/dp/1801070830/ref=sr_1_4?keywords=Machine+Learning+for+Imbalanced+Data&sr=8-4) today!

<a href="https://www.packtpub.com/product/machine-learning-for-imbalanced-data/9781801070836"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

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


### Related products <Other books you may enjoy>
* Machine Learning with PyTorch and Scikit-Learn  [[Packt]](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312) [[Amazon]](https://www.amazon.in/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319/ref=sr_1_3?keywords=Machine+Learning+with+PyTorch+and+Scikit-Learn&sr=8-3)
  
* Graph Machine Learning  [[Packt]](https://www.packtpub.com/product/graph-machine-learning/9781800204492) [[Amazon]](https://www.amazon.in/Graph-Machine-Learning-techniques-algorithms/dp/1800204493/ref=sr_1_3?keywords=Graph+Machine+Learning&sr=8-3)
  
## Get to Know the Author
**Kumar Abhishek** is a seasoned Senior Machine Learning Engineer at Expedia Group, US, specializing in risk analysis and fraud detection for Expedia brands. With over a decade of experience at companies such as Microsoft, Amazon, and a Bay Area startup, Kumar holds an MS in Computer Science from the University of Florida.

**Dr. Mounir Abdelaziz** is a deep learning researcher specializing in computer vision applications. He holds a Ph.D. in computer science and technology from Central South University, China. During his Ph.D. journey, he developed innovative algorithms to address practical computer vision challenges. He has also authored numerous research articles in the field of few-shot learning for image classification.


## Links

- [Amazon link](https://www.amazon.com/Machine-Learning-Imbalanced-Data-imbalanced/dp/1801070830/)

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
