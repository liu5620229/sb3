import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(num_points, sigma, upper, lower=0):
    upper = np.float32(upper)
    lower = np.float32(lower)
    increments = np.random.normal(0, sigma, num_points - 1).astype(np.float32)
    now = np.float32(np.random.uniform(lower, upper))
    trajectory = [now]

    for inc in increments:
        now += inc
        if now < lower:
            now = lower
        elif now > upper:
            now = upper
        trajectory.append(now)
    return trajectory
    # sigma设置为upper/100

MAX_D_i = 2.5

trajectory = np.array(brownian_motion(512, sigma=MAX_D_i / 10,
                                upper=MAX_D_i), dtype=np.float32)
plt.scatter(np.arange(0,512,1), trajectory)
plt.show()
