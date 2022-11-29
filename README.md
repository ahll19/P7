# P7

## Purpose of Repository
This repository is used for our 7th semester project. We are, unlike other semesters, supposed to write an article. The purpose of this article is to estimate mutual information (MI) between two two random variables. The data is created using the following generative model,

$$
    X_i\sim\mathcal{N}\left( 0,\ \sigma_xI_d \right), \\
    Y_i\sim\mathcal{N}\left( X_i,\ \sigma_yI_d \right).
$$

where $I_d$ is the identity matrix with dimension $d$. It should be noted that in the article $d=1$. The reason for using this generative model is that MI can be calculated as,

$$
    I(X_i;Y_i)=\frac{d}{2}\log_2\left( 1 + \frac{\sigma_x}{\sigma_y} \right).
$$

These MIs are used as labels to train a supervised learning recurrent neural network. 

