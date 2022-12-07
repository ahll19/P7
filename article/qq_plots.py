# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab 

def conf(x, theor, slope, confidence=0.95, dist="norm"):
    if isinstance(dist, str):
        dist = getattr(stats, dist)
    
    n = x.size
    P = (np.arange(n) + 1 - 0.5) / (n + 1 - 2 * 0.5)
    crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    #print(crit)
    #crit = 2.33

    fit_params = dist.fit(x)
    shape = fit_params[:-2] if len(fit_params) > 2 else None
    pdf = dist.pdf(theor) if shape is None else dist.pdf(theor, *shape)

    se = (slope / pdf) * np.sqrt(P * (1 - P) / n)
    upper = x + crit * se
    lower = x - crit * se
    
    return lower, upper


path_model_list = ['trained_models_FNN/06-12-2022_15-12-05', 'trained_models_CNN/06-12-2022_14-52-49', 
                'trained_models/21-11-2022_12-09-29']
# path = 'trained_models/21-11-2022_12-09-29/'
networks = ['FNN', 'CNN', 'RNN', 'KSG']

distribution = 'norm'

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


figsize = (10, 7)
fig, axes = plt.subplots(2, 2, sharex = True, figsize=figsize)
fig.suptitle('QQ Plot', fontsize=18)

y = [0,1,0,1]
x = [0,0,1,1]
for i, network in enumerate(networks):
    axes[x[i], y[i]].plot(theo[network][idx_inside[network]], data_response[network][idx_inside[network]], 
                    label='Inside ci', ls='None', marker='.', markersize=5)
    if idx_outside[network][0].size > 0:
        axes[x[i], y[i]].plot(theo[network][idx_outside[network]], data_response[network][idx_outside[network]], 
                        label='Outside ci', ls='None', marker='.', markersize=5)

    axes[x[i], y[i]].plot(theo[network], slope[network]*theo[network]+intercept[network], label='Reference\nLine', color='r')
    axes[x[i], y[i]].fill_between(theo[network], lower[network], upper[network], label='Confidence\nRegion', color='#ff00dd', alpha=0.1)

    if x[i] == 1:
        axes[x[i], y[i]].set_xlabel('Theoretical Quantiles')
    axes[x[i], y[i]].set_ylabel('Ordered Quantiles')

    axes[x[i], y[i]].grid()
    axes[x[i], y[i]].set_title(network, fontsize=12)
    
#axes.set_title(network, fontsize=12)
axes[0,1].legend(bbox_to_anchor=(1.55, 1.027), loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=12)
plt.tight_layout()
# plt.savefig('results/qq_model_residuals_ksg.pdf')
plt.show()
