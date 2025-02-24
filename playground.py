import numpy as np

a = 5
b = 3
l = [a,b]
n = np.array(l)

print(a, b, l, n)

l[0] = b

print(a, b, l, n)

