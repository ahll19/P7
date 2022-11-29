# P7 - Recurrent Neural Network for Estimating Mutual Information of a Gaussian Processes

## Purpose of Repository
This repository is used for our 7th semester project. We are, unlike other semesters, supposed to write an article. The purpose of this article is to estimate mutual information (MI) between two two random variables. The data is created using the following generative model,

$$ X_i\sim\mathcal{N}\left( 0,\ \sigma_xI_d \right), Y_i\sim\mathcal{N}\left( X_i,\ \sigma_yI_d \right). \tag{1}$$

where $I_d$ is the identity matrix with dimension $d$. It should be noted that in the article $d=1$. The reason for using this generative model is that MI can be calculated as [^fn1],

$$I(X_i;Y_i)=\frac{d}{2}\log_2\left( 1 + \frac{\sigma_x}{\sigma_y} \right). \tag{2}$$

These MIs are used as labels to train a supervised learning recurrent neural network (RNN). 

To test the RNN model we compare the performance to the KSG [^fn2] [^fn3].

## Usage
The code used for the article is found in the folder 'article'. A discription of each file from 'article' is found below:

`data_gen1.py`: This module uses (1) to generate the random variables $X$ and $Y$. Moreover, (2) is used to generate the MI between the two. 

`ksg.py`: This module uses `sklearn.feature_selection.mutual_info_regression()` to calculate the mutual information.

`model_test.py`, `models.py` and `train_test_v2.py`: `model_test.py` takes the trained model from `train_test_v2.py` and plots a summary. This is used to test different types of test data. `train_test_v2.py` is used to train a model and plot a summary. `models.py` is a module for `train_test_v2.py` with different functionalities. `qq_plots.ipynb` is used to test the distribution of the residuals using q-q plots. 



[^fn1]: “Statistical Inference of Information in Networks Causality and Directed Information Graphs kth royal institute of technology.” Accessed: Nov. 09, 2022. [Online]. Available: http://www.diva-portal.org/smash/get/diva2:1599246/FULLTEXT01.pdf
[^fn2]: A. Kraskov, H. Stögbauer, and P. Grassberger, “Estimating mutual information,” Physical Review E, vol. 69, no. 6, Jun. 2004, doi: 10.1103/physreve.69.066138.
[^fn3]: W. Gao, S. Oh, and P. Viswanath, “Demystifying Fixed k-Nearest Neighbor Information Estimators.” Accessed: Nov. 09, 2022. [Online]. Available: https://arxiv.org/pdf/1604.03006.pdf