import numpy as np
import matplotlib.pyplot as plt


def brownian_motion(num_points, sigma, upper, lower=0):
    upper = np.float32(upper)
    lower = np.float32(lower)
    increments = np.random.normal(0, sigma, num_points - 1).astype(np.float32)
    now = np.random.uniform(lower, upper)
    trajectory = [now]

    for inc in increments:
        now += inc
        if now < lower:
            now = lower
        elif now > upper:
            now = upper
        trajectory.append(now)
    return trajectory


trajectory = brownian_motion(512, sigma=20 / 20,
                       upper=20 / 4)
x = np.arange(0, 512, 1)

plt.scatter(x,trajectory)
plt.show()

