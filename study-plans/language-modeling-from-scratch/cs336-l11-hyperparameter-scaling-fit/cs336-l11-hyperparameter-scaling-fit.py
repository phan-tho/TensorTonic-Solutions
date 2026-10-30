import torch
import numpy as np

def llog(x):
    r = 0
    try:
        r = torch.log(x)
    except Exception as e:
        r = np.log(x)
    return r

def find_slop(ls, y):
    # given log scale and (log learning rate or log batch),
    # logy = a + b*log(ls)

    y = llog(y)
    x = llog(ls)

    N = x.shape[0]
    
    # Find coefficient a, b = argmin sum (y_i - a - b*x)^2
    s_x, s_x2, s_xy, s_y = 0,0,0,0
    for i in range(N):
        s_x += x[i]
        s_x2 += x[i]*x[i]
        s_xy += x[i] * y[i]
        s_y += y[i]
    s_x /= N
    s_x2 /= N
    s_xy /= N
    s_y /= N

    b = (s_x*s_y - s_xy) / (s_x*s_x - s_x2)
    a = s_y - b * s_x

    print("coef ", a.item(), b.item())

    return a.item(), b.item()

def c_residual(ls, truth, a, b):
    # given line y = a + b*ls
    # return an list (truth - y)

    pred = a + b * llog(ls)
    return (llog(truth) - pred)

def hyperparameter_scaling_fit(
    model_scales: torch.Tensor, learning_rates: torch.Tensor,
    batch_sizes: torch.Tensor, measured_losses: torch.Tensor,
    target_scale: int | float,
) -> dict:
    """
    Returns a dict of optima, scaling coefficients/exponents,
    residual tensors and target predictions.
    """

    N_LR: int = learning_rates.shape[0]
    N_B: int = batch_sizes.shape[0]

    optim_lr = [] 
    optim_b = []

    for i in range(measured_losses.shape[0]):
        # row are lr, column are b
        idx_min: int = measured_losses[i].argmin().item()
        
        optim_lr.append(learning_rates[idx_min // N_B].item())
        optim_b.append(batch_sizes[idx_min % N_B].item())

        # print(i, idx_min, learning_rates[idx_min // N_B].item(), batch_sizes[idx_min % N_B].item())
    optim_lr = torch.tensor(optim_lr)
    optim_b = torch.tensor(optim_b)
    
    a_lr, b_lr = find_slop(model_scales, optim_lr)
    a_b, b_b = find_slop(model_scales, optim_b)

    res_lr = c_residual(model_scales, optim_lr, a_lr, b_lr)
    res_b = c_residual(model_scales, optim_b, a_b, b_b)

    lr_pred = np.exp(a_lr + b_lr * llog(target_scale))
    b_pred = np.exp(a_b + b_b * llog(target_scale))

    print("predict ", lr_pred, b_pred)

    ans = {"optima": []}

    # ans["optima"] = []
    for i in range(model_scales.shape[0]):
        d_i = {"model_scale": model_scales[i].item(),
               "learning_rate": optim_lr[i].item(), 
               "batch_size": optim_b[i].item(), 
               "loss": measured_losses[i].min().item()}
        ans["optima"].append(d_i)
    ans["learning_rate_coefficient"] = np.exp(a_lr).item()
    ans["learning_rate_exponent"] = b_lr
    ans["batch_size_coefficient"] = np.exp(a_b).item()
    ans["batch_size_exponent"] = b_b
    ans["learning_rate_residuals"] = c_residual(model_scales, optim_lr, a_lr, b_lr)
    ans["batch_size_residuals"] = c_residual(model_scales, optim_b, a_b, b_b)
    ans["target_learning_rate"] = lr_pred.item()
    ans["target_batch_size"] = b_pred.item()

    print(ans)

    return ans
