# P7 - Recurrent Neural Network for Estimating Mutual Information of a Gaussian Processes

## Purpose of Repository
This repository is used for our 7th semester project. We are, unlike other semesters, supposed to write an article. The purpose of this article is to estimate mutual information (MI) between two two random variables. The data is created using the following generative model,

$$
    X_i\sim\mathcal{N}\left( 0,\ \sigma_xI_d \right), \\
    Y_i\sim\mathcal{N}\left( X_i,\ \sigma_yI_d \right).
$$

where $I_d$ is the identity matrix with dimension $d$. It should be noted that in the article $d=1$. The reason for using this generative model is that MI can be calculated as [^fn1],

\begin{equation}
    I(X_i;Y_i)=\frac{d}{2}\log_2\left( 1 + \frac{\sigma_x}{\sigma_y} \right). \label{eq:mutual_inf}
\end{equation}

These MIs are used as labels to train a supervised learning recurrent neural network. 

## Usage
The code used for the article is found in the folder 'article'. A discription of each file from 'article' is found below:

`data_gen1.py`: This module uses \eqref{eq:mutual_inf} to 



[^fn1]: “Statistical Inference of Information in Networks Causality and Directed Information Graphs kth royal institute of technology.” Accessed: Nov. 09, 2022. [Online]. Available: http://www.diva-portal.org/smash/get/diva2:1599246/FULLTEXT01.pdf