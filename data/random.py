import numpy as np
import matplotlib.pyplot as plt

data1 = np.random.normal(1.75, 0.1, 1000000)
print(data1.shape)

plt.figure(figsize=(10, 2*4), dpi=80)

plt.subplot(2, 1, 1)
plt.hist(data1, bins=1000)

data2 = np.random.uniform(-1, 1, 1000000)
print(data2.shape)

plt.subplot(2, 1, 2)
plt.hist(data2, bins=1000)

plt.show()
