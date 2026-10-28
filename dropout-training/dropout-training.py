import numpy as np

# idx = np.random.random(10)
# mask = (idx < 0.3)
# print(idx)
# print(mask)
# a = mask * 2.0
# print(a * a)

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.array(x)
    idx = np.random.random(x.shape)
    if rng:
        idx = rng.random(x.shape)
    mask = (idx > p) / (1 - p)
    x = mask * x
    return (x, mask)