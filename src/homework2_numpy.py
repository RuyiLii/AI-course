import numpy as np
import torch
import matplotlib.pyplot as plt
import math

x = np.linspace(-math.pi,math.pi,2000)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learn_rating = 1e-6

for t in range(5000):
    y_pred = a + b*x + c*x**2+d*x**3
    loss = ((y_pred - y)**2).sum()
    if t % 100 == 99:
        print(t,loss)
    
    grad_y_pred = 2*(y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred*x).sum()
    grad_c = (grad_y_pred*x**2).sum()
    grad_d = (grad_y_pred*x**3).sum()

   
    a-= learn_rating*grad_a
    b-= learn_rating*grad_b
    c-= learn_rating*grad_c
    d-= learn_rating*grad_d

print(f'result: y ={a} + {b}x + {c}x^2 + {d}x^3')

plt.plot(x,y)
plt.plot(x,y_pred)
plt.legend(['y=sin(x)','y=y_pred'],loc="upper left")
plt.show()
