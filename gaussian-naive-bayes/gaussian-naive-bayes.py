import math
import numpy as np

def llog(a):
    return np.log(a + 1e-9)

def prob_gau(x, mu=0, s2=1):
    # logP_N (x)
    return -1/2 * llog(2*math.pi*s2) - ((x - mu)**2)/(2*s2 + 2e-9)

def prepare(X_train, y_train):
    N_CLASS = y_train.max() + 1
    N_FEAT = X_train.shape[1]

    prob_y = np.zeros(N_CLASS)
    mu = np.zeros(shape=(N_FEAT, N_CLASS))   # \mu[k][j] feat k, class j
    s2 = np.zeros(shape=(N_FEAT, N_CLASS))

    # print(mu.shape)

    for j in range(N_CLASS):
        # if prob_y = 0
        
        # print(X_train[y_train == j])
        # X_train[y_train == j] shape = N, N_FEAT
        mask = y_train == j
        
        # print(mask, mask.sum(), mask.size)
        
        prob_y[j] = mask.sum() / mask.size
        
        x_c = X_train[mask]
        mu[:, j] = x_c.mean(0)
        s2[:, j] = x_c.var(0)

        # print(prob_y[j])
        # print(x_c)
        # print(x_c.mean(0))
        # print(x_c.var(0))
    
    return prob_y, mu, s2
    

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    # Write code here
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    N_CLASS = y_train.max() + 1
    N_FEAT = X_train.shape[1]
    
    y_pred = np.zeros(X_test.shape[0], dtype=int)

    prob_y, mu, s2 = prepare(X_train, y_train)

    for i in range(X_test.shape[0]):
        # X_test[i]: array shape N_FEAT
        
        max_prob = -1e18
        pred_idx = 0

        for j in range(N_CLASS):
            # calculate logP(Y = j|X_1 = X_test[i][1], ....)
            # = logP(Y = j) + \sum_k logP(X_k = X_test[i][k] | Y = j)
            # where X_k | Y = j \sim N(\mu_{kj}, \sigma_{kj}^2)

            print("class", j)
            
            prob = llog(prob_y[j])
            for k in range(N_FEAT):
                prob += prob_gau(X_test[i][k], mu[k][j], s2[k][j])

            print(prob, max_prob)
            
            if prob > max_prob:
                max_prob = prob
                pred_idx = j
                print(pred_idx)
        y_pred[i] = pred_idx

    # print(y_pred)
    # print(y_pred.tolist())
        
    return y_pred.tolist()