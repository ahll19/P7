import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab 

def conf(x, theor, slope, confidence=0.95, dist="norm"):
    """
    Compute the confidence interval of a given set of data.
    
    This function calculates the confidence interval of a given set of data, using
    the specified probability distribution, slope, and confidence level.
    
    Args:
        x (array-like): The set of data to compute the confidence interval for.
        theor (array-like): The theoretical value to use in the computation.
        slope (float): The slope of the line to use in the computation.
        confidence (float, optional): The confidence level to use, as a fraction between 0 and 1. Defaults to 0.95.
        dist (str or stats.distribution, optional): The probability distribution to use. This can either be a string
            representing a known distribution (e.g. 'norm' for the normal distribution), or a stats.distribution object.
            Defaults to 'norm'.
    
    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    if isinstance(dist, str):
        dist = getattr(stats, dist)
    
    n = x.size
    P = (np.arange(n) + 1 - 0.5) / (n + 1 - 2 * 0.5)
    crit = stats.norm.ppf(1 - (1 - confidence) / 2)

    fit_params = dist.fit(x)
    shape = fit_params[:-2] if len(fit_params) > 2 else None
    pdf = dist.pdf(theor) if shape is None else dist.pdf(theor, *shape)

    se = (slope / pdf) * np.sqrt(P * (1 - P) / n)
    upper = x + crit * se
    lower = x - crit * se
    
    return lower, upper

def getQQData(path_model_list, networks, distribution = 'norm'):
    """
    Get data for QQ plots.
    
    This function computes the data necessary to create QQ plots for the given networks and models, using the
    specified probability distribution.
    
    Args:
        path_model_list (list): A list of paths to the models to use in the computation.
        networks (list): A list of networks to compute data for.
        distribution (str, optional): The probability distribution to use. Defaults to 'norm'.
    
    Returns:
        dict: A dictionary containing the computed data. This will include the following keys:
            'model_out': The model outputs.
            'ksg': The known-shape Gaussians.
            'label_tst': The test labels.
            'model_err': The model errors.
            'ksg_err': The known-shape Gaussian errors.
            'theo': The theoretical quantiles.
            'data_response': The data response.
            'slope': The slope of the reference line.
            'intercept': The intercept of the reference line.
            'lower': The lower bound of the confidence interval.
            'upper': The upper bound of the confidence interval.
            'idx_inside': The indices of the data points inside the confidence interval.
            'idx_outside': The indices of the data points outside the confidence interval.
    """


    model_out = {}
    ksg = {}
    label_tst = {}
    model_err = {}
    ksg_err = {}
    theo = {}
    data_response = {}
    slope = {}
    intercept = {}
    lower = {}
    upper = {}
    idx_inside = {}
    idx_outside = {}
    for key, path in enumerate(path_model_list):
        network = networks[key]
        with open(path + '/model_out-ksg-label_tst.npy', 'rb') as f:
            model_out[network] = np.load(f)
            ksg['KSG'] = np.load(f)
            label_tst[network] = np.load(f)

        model_err[network] = model_out[network] - label_tst[network]

        (theo[network], data_response[network]), (slope[network], intercept[network], r) = \
            stats.probplot(model_err[network].T[0], dist=distribution)

        lower[network], upper[network] = conf(slope[network]*theo[network]+intercept[network], 
                                                            theo[network], slope[network], 
                                                            confidence=0.95, dist=distribution)

        idx_inside[network] = np.where((lower[network] <= data_response[network]) &
                                                (data_response[network] <= upper[network]))
        idx_outside[network] = np.where((lower[network] > data_response[network]) |
                                                (data_response[network] > upper[network]))

    ksg_err['KSG'] = ksg['KSG'] - label_tst[network]

    (theo['KSG'], data_response['KSG']), (slope['KSG'], intercept['KSG'], r) = \
        stats.probplot(ksg_err['KSG'].T[0], dist=distribution)

    lower['KSG'], upper['KSG'] = conf(slope['KSG']*theo['KSG']+intercept['KSG'], 
                                                        theo['KSG'], slope['KSG'], 
                                                        confidence=0.95, dist=distribution)

    idx_inside['KSG'] = np.where((lower['KSG'] <= data_response['KSG']) &
                                            (data_response['KSG'] <= upper['KSG']))
    idx_outside['KSG'] = np.where((lower['KSG'] > data_response['KSG']) |
                                            (data_response['KSG'] > upper['KSG']))
    
    data = {
        'model_out': model_out,
        'ksg': ksg,
        'label_tst': label_tst,
        'model_err': model_err,
        'ksg_err': ksg_err,
        'theo': theo,
        'data_response': data_response,
        'slope': slope,
        'intercept': intercept,
        'lower': lower,
        'upper': upper,
        'idx_inside': idx_inside,
        'idx_outside': idx_outside
    }
    return data


def createPlots(data, networks):
    """
    Create plots of the given data for each network.
    
    This function creates plots of the given data for each network, using the provided data to plot 
    the theoretical quantiles, data response, confidence interval, and reference line for each network.
    
    Args:
        data (dict): A dictionary containing the data to use in the plots. This should include the following keys:
            'theo': The theoretical quantiles to use in the plot.
            'data_response': The data response to use in the plot.
            'slope': The slope of the reference line.
            'intercept': The intercept of the reference line.
            'lower': The lower bound of the confidence interval.
            'upper': The upper bound of the confidence interval.
            'idx_inside': The indices of the data points inside the confidence interval.
            'idx_outside': The indices of the data points outside the confidence interval.
        networks (list): A list of networks to create plots for.
    
    Returns:
        None
    """

    theo = data['theo']
    data_response = data['data_response']
    slope = data['slope']
    intercept = data['intercept']
    lower = data['lower']
    upper = data['upper']
    idx_inside = data['idx_inside']
    idx_outside = data['idx_outside']

    figsize = (8, 6)
    fig, axes = plt.subplots(1, len(networks), sharex = True, figsize=figsize)
    # fig.suptitle('QQ Plot - Gaussian', fontsize=18)

    y = [0,1,0,1]
    x = [0,0,1,1]
    # for i, network in enumerate(networks):
    #     axes[x[i], y[i]].plot(theo[network][idx_inside[network]], data_response[network][idx_inside[network]], 
    #                     label='Inside ci', ls='None', marker='.', markersize=5)
    #     if idx_outside[network][0].size > 0:
    #         axes[x[i], y[i]].plot(theo[network][idx_outside[network]], data_response[network][idx_outside[network]], 
    #                         label='Outside ci', ls='None', marker='.', markersize=5)
    #     axes[x[i], y[i]].plot(theo[network], slope[network]*theo[network]+intercept[network], label='Reference\nLine', color='r')
    #     axes[x[i], y[i]].fill_between(theo[network], lower[network], upper[network], label='Confidence\nRegion', color='#ff00dd', alpha=0.1)
    #     if x[i] == 1:
    #         axes[x[i], y[i]].set_xlabel('Theoretical Quantiles')
    #     axes[x[i], y[i]].set_ylabel('Ordered Quantiles')
    #     axes[x[i], y[i]].grid()
    #     axes[x[i], y[i]].set_title(network, fontsize=12)
    if len(networks) == 1:
        network = networks[0]
        axes.plot(theo[network][idx_inside[network]], data_response[network][idx_inside[network]], 
                        label='Inside ci', ls='None', marker='.', markersize=5)
        if idx_outside[network][0].size > 0:
            axes.plot(theo[network][idx_outside[network]], data_response[network][idx_outside[network]], 
                            label='Outside ci', ls='None', marker='.', markersize=5)
        axes.plot(theo[network], slope[network]*theo[network]+intercept[network], label='Reference Line', color='r')
        axes.fill_between(theo[network], lower[network], upper[network], label='Confidence Region', color='#ff00dd', alpha=0.1)
        axes.set_xlabel('Theoretical Quantiles', size=15)
        axes.set_ylabel('Ordered Quantiles', size=15)
        axes.grid()
        # axes.set_title(network, fontsize=15)
        axes.yaxis.set_tick_params(labelsize=15)
        axes.xaxis.set_tick_params(labelsize=15)
        axes.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", fancybox=True, shadow=True, ncol=2, fontsize=15)
    else:
        for i, network in enumerate(networks):
            axes[i].plot(theo[network][idx_inside[network]], data_response[network][idx_inside[network]], 
                            label='Inside ci', ls='None', marker='.', markersize=5)
            if idx_outside[network][0].size > 0:
                axes[i].plot(theo[network][idx_outside[network]], data_response[network][idx_outside[network]], 
                                label='Outside ci', ls='None', marker='.', markersize=5)
            axes[i].plot(theo[network], slope[network]*theo[network]+intercept[network], label='Reference\nLine', color='r')
            axes[i].fill_between(theo[network], lower[network], upper[network], label='Confidence\nRegion', color='#ff00dd', alpha=0.1)
            axes[i].set_xlabel('Theoretical Quantiles')
            axes[i].set_ylabel('Ordered Quantiles')
            axes[i].grid()
            axes[i].set_title(network, fontsize=15)
        axes[1].legend(bbox_to_anchor=(1.55, 1.027), loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=15)
    plt.tight_layout()
    # plt.savefig('results/qq_model_residuals_ksg.pdf')
    plt.show()

if __name__ == "__main__":

    ############### Input ###############
    model = 'RNN' # KSG if qq-plot for KSG residuals. RNN if qq-plot for network residuals

    ############### Run script ###############
    networks = [model]
    path_model_list = ['trained_models/21-11-2022_12-09-29']
    qq_data = getQQData(path_model_list, networks)
    createPlots(qq_data, networks)