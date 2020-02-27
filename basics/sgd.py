'''
http://zh.gluon.ai/chapter_optimization/gd-sgd.html

梯度下降（gradient descent）的工作原理
'''
import numpy as np
import matplotlib.pyplot as plt

def grad(start, eta, epochs):
    x = start
    results = [x]
    for i in range(epochs):
        x -= eta * 2 * x   # f'(x) = 2 * x for x^2
        results.append(x)

    print('epoch {}, x:{}'.format(epochs, x))
    return results

def show_trace(res):
    n = max(abs(min(res)), abs(max(res)))
    f_line = np.arange(-n, n, 0.1)

    plt.plot(f_line, [x * x for x in f_line])
    plt.plot(res, [x * x for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def loss_function_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2

def grads_2d(x1, x2, s1, s2, eta):
    x1_grad = eta * 2 * x1
    x2_grad = eta * 4 * x2

    return (x1 - x1_grad, x2 - x2_grad, 0, 0)

def train_2d(x1, x2, eta, trainer, epochs):
    s1, s2 = 0, 0
    results = [(x1, x2)]
    for i in range(epochs):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2, eta)
        results.append((x1, x2))

    print('epochs {} gets x1: {}, x2: {}'.format(epochs, x1, x2))
    return results

def show_trace_2d(f, results):

    plt.plot(*zip(*results), '-o', color='#ff7f0e')

    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

if __name__ == '__main__':
    res = grad(10, 0.2, 100)
    show_trace(res)

    results = train_2d(-5, -2, 0.1, grads_2d, 20)
    show_trace_2d(loss_function_2d, results)

