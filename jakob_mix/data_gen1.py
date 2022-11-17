#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mn

def gen_list(n, var_x, var_y, dim=2):
    """
    Generates a set of random points according to <statistical inference of information in networks>
    on page 51. The third variable, Z, is ignored as this is only used to create training data
    for MI, not CMI.
    The covariance matrices are both a scaled identity matrix.
    :param n: Number of points realized
    :param var_x: Variance of X
    :param var_y: Variance of Y
    :param dim: Dimension of the variables X and Y
    :return:
    :returns values: a realization of the process described in the book. Dimensions are [n, X/Y, dim].
                     e.g. [20, 0, 2] gives the second element the 20th realization of X.
    :returns mi: The mutual information: I(X; Y)
    """
    values = np.zeros((n, 2, dim))  # init
    values[:, 0, :] = mn.rvs(mean=np.zeros(dim), cov=var_x * np.identity(dim), size=(n, dim)).reshape((n, dim))

    for i, x in enumerate(values[:, 0, :]):
        values[i, 1, :] = mn.rvs(mean=x, cov=var_y * np.identity(dim), size=1)

    mi = dim * np.log2(1 + var_x / var_y) / 2

    return values, mi


def create_data(realisations, xy_len=1000, min_var0=0.1,
                 max_var0=10, min_var1=0.1, max_var1=10, dim=1):
    
    X_var = np.random.uniform(min_var0, max_var0, realisations)
    Y_var = np.random.uniform(min_var1, max_var1, realisations)
    data = np.zeros((realisations, xy_len, 2))
    label = np.zeros([realisations])

    for i in range(realisations):
        _test, label[i] = gen_list(
            xy_len, X_var[i], Y_var[i], dim)
        x, y = _test[:, 0, :], _test[:, 1, :]

        data[i] = np.hstack((x, y))
    
    return data, label, X_var, Y_var


def save_data(file_path, data, labels):
    """
    Takes the path and name in one string and saves the file
    at that location as a .npy.

    :param file_path: The path with the path to the folder where the
    data is saved.
    :param data: The samples consisting of x and y
    :param labels: Labels to the data
    :return: No returns

    Example using savedata/loaddata:
    path = "test"
    save_data(path, data, labels) # saves data as .npy file
    data, labels = load_data(path) # loads .npy file
    """
    if file_path[-4:] != '.npy':
        file_path += '.npy'

    with open(file_path, 'wb') as f:
        np.save(f, data)
        np.save(f, labels)


def load_data(file_path):
    """
    Takes the path to where the .npy file is located.

    :param file_path: The path with the path to the folder where the
    data is saved.
    :return: data, labels

    Example using savedata/loaddata:
    path = "test"
    save_data(path, data, labels) # saves data as .npy file
    data, labels = load_data(path) # loads .npy file
    """
    if file_path[-4:] != '.npy':
        file_path += '.npy'

    with open(file_path, 'rb') as f:
        data = np.load(f)
        label = np.load(f)

    return data, label


if __name__ == "__main__":
    realisations = 1000
    xy_len = 50
    '''
    For high AVG + low min + high max
    Max MI: 5.48325
    Min MI: 0.00360
    Mean:   1.79069
    min_var_x = 0.05
    max_var_x = 100
    min_var_y = 0.05
    max_var_y = 10

    Lower version of above
    Max MI: 4.32373
    Min MI: 0.00360
    Mean:   0.88200
    min_var_x = 0.05
    max_var_x = 20
    min_var_y = 0.05
    max_var_y = 10

    For Outliers with high MI
    Max MI: 5.48325
    Min MI: 0.00036
    Mean:   0.65730
    min_var_x = 0.05
    max_var_x = 100
    min_var_y = 0.05
    max_var_y = 100

    Mostly zeros 
    Max MI: 3.82553
    Min MI: 0.00036
    Mean:   0.15158
    min_var_x = 0.05
    max_var_x = 10
    min_var_y = 0.05
    max_var_y = 100

    A little better than above
    Max MI: 3.82553
    Min MI: 0.00180
    Mean:   0.42225
    min_var_x = 0.05
    max_var_x = 10
    min_var_y = 0.05
    max_var_y = 20

    Old time fav 
    Max MI: 3.32911
    Min MI: 0.00718
    Mean:   0.60823
    min_var_x = 0.1
    max_var_x = 10
    min_var_y = 0.1
    max_var_y = 10

    High MI overall
    Max MI: 5.48325
    Min MI: 0.50000
    Mean:   4.51636
    min_var_x = 0.1
    max_var_x = 100
    min_var_y = 0.05
    max_var_y = 0.1
    '''

    min_var_x = np.random.uniform(0.01, 2)
    max_var_x = min_var_x + np.random.uniform(0.01, 100)
    min_var_y = np.random.uniform(0.01, 2)
    max_var_y = min_var_y + np.random.uniform(0.01, 100)

    data, labels, X_var, Y_var = create_data(
        realisations, 
        xy_len,
        min_var_x,
        max_var_x,
        min_var_y,
        max_var_y
    )

    print(f"""
    Max MI: {(1/2)*np.log2(1+(max_var_x/min_var_y)):.5f}
    Min MI: {(1/2)*np.log2(1+(min_var_x/max_var_y)):.5f}
    Mean:   {np.mean(labels):.5f}
    min_var_x = {min_var_x}
    max_var_x = {max_var_x}
    min_var_y = {min_var_y}
    max_var_y = {max_var_y}
     """)

    plt.plot(labels, '*')
    plt.show()

    plt.hist(labels)
    plt.show()
    
    plt.plot(data[0,:,:])
    plt.title(f'MI: {labels[0]}')
    plt.show()

    # path = f"data/({realisations},{xy_len},2)_xvar-{min_var_x}-{max_var_x}_yvar-{min_var_y}-{max_var_y}"
    # save_data(path, data, labels)
    # data_loaded, labels_loaded = load_data(path)
    #%%
    if True:
        np.random.seed(69)
        #realisations = 1000
        xy_len = 50
        maxiter = 1000000
        max_MI = 2.5
        inters = 10
        dat_pr_interval = 10
        max_var = 100
        min_var_lower = 0.01
        min_var_upper = 2
        max_var_lower = 0.01
        max_var_upper = 100
        save = True

        intervals = np.linspace(0, max_MI, inters)
        counter = np.zeros(len(intervals)-1)
        data_len = int((inters-1)*dat_pr_interval)
        dat_unif = np.zeros((data_len, xy_len, 2))
        lab_unif = np.zeros(data_len)
        saved_idx = 0
        variance = np.zeros((data_len, 2))
        status = 0
        for i in range(maxiter):
            min_var_x = np.random.uniform(min_var_lower, min_var_upper)
            max_var_x = min_var_x + np.random.uniform(max_var_lower, max_var_upper)
            min_var_y = np.random.uniform(min_var_lower, min_var_upper)
            max_var_y = min_var_y + np.random.uniform(max_var_lower, max_var_upper)

            data, labels, X_var, Y_var = create_data(
                1, 
                xy_len,
                min_var_x,
                max_var_x,
                min_var_y,
                max_var_y
            )

            if labels[0]>=max_MI:
                continue
            if labels[0] == 0:
                continue

            count_idx = np.searchsorted(intervals, labels[0])-1

            if counter[count_idx] < dat_pr_interval:
                dat_unif[saved_idx] = data
                lab_unif[saved_idx] = labels[0]
                variance[saved_idx] = np.concatenate([X_var, Y_var])
                saved_idx += 1
                counter[count_idx] += 1

                if counter[count_idx] == dat_pr_interval:
                    status += 1
                    print(f'Status {status}/{inters}  -  {intervals[count_idx]:.5f}-{intervals[count_idx+1]:.5f}')

            if np.all(counter == np.full(len(counter), dat_pr_interval)):
                break
        
        
        # shuffle
        idx = np.random.choice(len(lab_unif), len(lab_unif), replace=False)
        dat_unif = dat_unif[idx]
        lab_unif = lab_unif[idx]
        variance = variance[idx]

        plt.plot(lab_unif, '*')
        plt.show()

        plt.hist(lab_unif, bins=len(counter))
        plt.show()

        plt.hist(variance[:,0], label='X var', alpha=0.5)
        plt.hist(variance[:,1], label='Y var', alpha=0.5)
        plt.legend()
        plt.show()
        
        if save:
            import pandas as pd
            from datetime import datetime

            now = datetime.now()
            time = now.strftime("%d-%m-%Y_%H-%M-%S")

            sample_dict = {
                'xy_len' : xy_len,
                'Max MI' : max_MI,
                'Amount of intervals' : inters-1,
                'Data per interval' : dat_pr_interval,
                'Data length' : data_len,
                'min_var_lower': min_var_lower,
                'min_var_upper': min_var_upper,
                'max_var_lower': max_var_lower,
                'max_var_upper': max_var_upper
            }
            df = pd.DataFrame.from_dict([sample_dict]) 
            df.to_csv (f'data/{time}.csv', index = False, header=True)

            path = f"data/{time}_unif_{len(dat_unif)}"
            save_data(path, dat_unif, lab_unif)

            with open(f'data/{time}_var_{len(dat_unif)}.npy', 'wb') as f:
                np.save(f, variance)
            

# %%
