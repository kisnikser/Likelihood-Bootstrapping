import numpy as np
from tqdm.auto import tqdm
from models import RegressionModel, LogisticModel
from data import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

def D(means, variances):
    return variances

def M(means, variances):
    return np.abs(np.diff(means, n=1))

def get_means_variances(X, y, sample_sizes=None, task="regression", sigma2=1, B=100):
    if sample_sizes is None:
        sample_sizes = np.arange(X.shape[1]+1, X.shape[0])
        
    means = []
    variances = []
    
    dataset = Dataset(X, y, task)
    
    if task == "regression":
        Model = RegressionModel
    elif task == "classification":
        Model = LogisticModel
    elif task == "liver-disorders":
        loss = mean_squared_error
        Model = LinearRegression()
    else:
        raise NotImplementedError

    for k in tqdm(sample_sizes):
        tmp = []
        for _ in range(B):
            X_k, y_k = dataset.sample(k)
            if task == "regression":  
                model = Model(X_k, y_k)
                w_hat = model.fit()
                tmp.append(model.loglikelihood(w_hat, X, y, sigma2))
            elif task == "classification":
                model = Model(X_k, y_k)
                w_hat = model.fit()
                tmp.append(model.loglikelihood(w_hat, X, y))
            else:
                Model.fit(X_k, y_k)
                y_pred = Model.predict(X)
                tmp.append(loss(y, y_pred))
        tmp = np.array(tmp)
        means.append(tmp.mean())
        variances.append(tmp.var())
        
    means = np.array(means)
    variances = np.array(variances)
    
    return means, variances


def sufficient_sample_size(sample_sizes: np.ndarray,
                           means: np.ndarray = None,
                           variances: np.ndarray = None,
                           eps=1e-4, 
                           method="variance"):
    """
    Calculate sufficient sample size. Use method with threshold eps.
    """
    
    if method not in ["variance", "rate"]:
        raise NotImplementedError

    if method == 'variance' and variances is None:
        return ValueError
    
    if method == 'rate' and means is None:
        return ValueError

    m_star = np.inf
        
    if method == "variance":
        for k, var in zip(sample_sizes, D(means, variances)):
            if var <= eps and m_star == np.inf:
                m_star = k
            elif var > eps:
                m_star = np.inf
                
    elif method == "rate":
        for k, diff in zip(sample_sizes[:-1], M(means, variances)):
            if diff <= eps and m_star == np.inf:
                m_star = k
            elif diff > eps:
                m_star = np.inf
        
    return m_star


def sufficient_vs_threshold(sample_sizes: np.ndarray,
                            means: np.ndarray,
                            variances: np.ndarray,
                            thresholds: np.ndarray):
    """
    Calculate sufficient sample sizes for each eps in thresholds.
    """
    sufficient = {'variance': [],
                  'rate': []}
    
    for method in ['variance', 'rate']:
        for eps in thresholds:
            sufficient[method].append(sufficient_sample_size(sample_sizes=sample_sizes,
                                                            means=means,
                                                            variances=variances,
                                                            eps=eps,
                                                            method=method))
    
    return sufficient
    
    
####### APPROXIMATION ########

    
def func_mean_approx(k, w):
    #return w[0] - (w[1]**2) * np.exp((w[2]**2) * k) - (w[3]**2) / (k**1.5)
    return w[0] + w[1] * np.exp(w[2] * k)

def func_var_approx(k, w):
    #return w[0] - (w[1]**2) * np.exp((w[2]**2) * k) - (w[3]**2) / (k**1.5)
    return w[0] + w[1] * np.exp(w[2] * k)

def approx(sample_sizes,
           means,
           variances,
           func_mean=func_mean_approx, 
           func_var=func_var_approx, 
           n_means=3, 
           n_variances=3,
           w0_means=None, 
           w0_variances=None, 
           train_size=0.5, verbose=False):
    
    # initial point for optimizing parameters w
    #w0_means = np.random.normal(size=n_means) if w0_means is None else w0_means
    #w0_variances = np.random.normal(size=n_variances) if w0_variances is None else w0_variances
    w0_means = np.zeros(n_means) if w0_means is None else w0_means
    w0_variances = np.zeros(n_variances) if w0_variances is None else w0_variances
    # number of points in train sample
    M = int(train_size*sample_sizes.size)
        
    X_train_means = sample_sizes[:M]
    y_train_means = means[:M]
    
    X_train_variances = sample_sizes[:M]
    y_train_variances = variances[:M]
        
    # find parameters w, that minimize MSE between log-likelihood (-loss) mean and it's approximation
    # start optimizing from w = w0
    means_minimum = minimize(lambda w: ((func_mean(X_train_means, w) - y_train_means)**2).mean(), w0_means)
    w_means = means_minimum.x
    variances_minimum = minimize(lambda w: ((func_var(X_train_variances, w) - y_train_variances)**2).mean(), w0_variances)
    w_variances = variances_minimum.x
    
    means_approximation = func_mean(sample_sizes, w_means)
    variances_approximation = func_var(sample_sizes, w_variances)

    return means_approximation, variances_approximation