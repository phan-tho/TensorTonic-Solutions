import torch
import numpy as np

def llog(x):
    r = 0
    try:
        r = torch.log(x)
    except Exception as e:
        r = np.log(x)
    return r

def find_slop(x, y, f):
    # given log scale and (log learning rate or log batch),
    # logy = a + b*log(ls)

    y = llog(y - f)
    x = llog(x)

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


def candidate(floor, x, y):
    """
    Return a, b of line and RMSE of this line in original scale
    """
    a, b = find_slop(x, y, floor)

    pred = floor + np.exp(a + b*llog(x))
    
    pred -= y

    print("pred", pred)
    
    rmse = np.sqrt(np.inner(pred, pred)/x.shape[0])
    
    return a, b, rmse

def fit_data_power_law(data_sizes: torch.Tensor, losses: torch.Tensor, target_sizes: torch.Tensor, floor_candidates: torch.Tensor) -> dict:
    """
    Returns a dict: floor, coefficient, exponent, predictions (float64 tensor), rmse.
    """
    min_err: float = 1e9
    a, b, floor = 0.0, 0.0, 0.0
    for f in floor_candidates:
        _a, _b, err = candidate(f, data_sizes, losses)
        if err < min_err:
            min_err = err
            a = _a
            b = _b
            floor = f
    
    ans = {"floor": floor.item(), "coefficient": np.exp(a).item(), "exponent": -b}
    pred = floor + np.exp(a + b*llog(target_sizes))
    ans["predictions"] = pred
    ans["rmse"] = min_err.item()

    print(ans)
        
    return ans
