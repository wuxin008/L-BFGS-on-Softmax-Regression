from pathlib import Path
dir_name = Path(__file__).parent
import os
os.chdir(dir_name)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import *
from l_bfgs import l_bfgs

def load_data(name: str) -> List[np.ndarray]:
    train_path = f'{name}/train'
    test_path = f'{name}/test'

    res = []
    for path in [train_path, test_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f'{path} not found')
        
        x_data = np.loadtxt(f'{path}/x.txt')
        y_data = np.loadtxt(f'{path}/y.txt')

        if x_data.shape[0]!= y_data.shape[0]:
            raise ValueError(f'x and y data have different lengths: {x_data.shape[0]}!= {y_data.shape[0]}')
        
        res.append(np.hstack((x_data, y_data.reshape(-1, 1))))

    return res

def plot_data(loss: List[float], thetas: np.ndarray, data: np.ndarray, title=None,binary=True) -> None:
    x_train = data[:, :-1]
    y_train = data[:, -1]

    fig = plt.figure(figsize=(12, 6))
    if title is not None:
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.subplot(1, 2, 1)
    loss_line, *_ = plt.plot(np.arange(len(loss)), loss, '-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    if binary == True:
        x_data = np.linspace(np.min(x_train[:, 1]), np.max(x_train[:, 1]), 100)
        y_data = -thetas[0][1] / thetas[0][2] * x_data - thetas[0][0] / thetas[0][2]
        separator, *_ = plt.plot(x_data, y_data, '-')

        def update(frame: int) -> tuple[object]:
            loss_line.set_data(np.arange(frame), loss[:frame])
            separator.set_data(x_data, -thetas[frame][1] / thetas[frame][2] * x_data - thetas[frame][0] / thetas[frame][2])
            return loss_line, separator, scat
    else:
        x_data = np.linspace(np.min(x_train[:, 1]) - 2, np.max(x_train[:, 1]) + 2, 50)
        y_data = np.linspace(np.min(x_train[:, 1]) - 2, np.max(x_train[:, 1]) + 2, 50)
        xx, yy = np.meshgrid(x_data, y_data)
        z = np.hstack((np.ones((xx.ravel().shape[0], 1)), np.vstack([xx.ravel(), yy.ravel()]).T))
        k = np.argmax(np.exp(z @ thetas[0]) / np.sum(np.exp(z @ thetas[0]), axis=1).reshape(-1, 1), axis=1).reshape(-1)
        colors = (k - np.min(k)) / (np.max(k) - np.min(k))
        classes = plt.scatter(xx.ravel(), yy.ravel(), c=colors, cmap='plasma')

        def update(frame: int) -> tuple[object]:
            frame = frame * 5
            loss_line.set_data(np.arange(frame), loss[:frame])
            k = np.argmax(np.exp(z @ thetas[frame]) / np.sum(np.exp(z @ thetas[frame]), axis=1).reshape(-1, 1), axis=1).reshape(-1)
            colors = (k - np.min(k)) / (np.max(k) - np.min(k))
            classes.set_array(colors)
            return loss_line, classes, scat
        
    scat = plt.scatter(x_train[:, 1], x_train[:, 2], c=y_train.reshape(-1))
    plt.xlim(np.min(x_train[:, 1] - 5), np.max(x_train[:, 1] + 5))
    plt.ylim(np.min(x_train[:, 2] - 5), np.max(x_train[:, 2] + 5))

    anim = FuncAnimation(fig=fig, func=update, frames=np.arange(len(loss) // 5), interval=1, blit=True, repeat=False)
    anim.save(title + '.gif', writer='ffmpeg', fps=60)
    plt.show()

def softmax(X: np.ndarray) -> np.ndarray:
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)

def g_softmax(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, classes: int) -> np.ndarray:
    mask = np.zeros((X.shape[0], classes))
    mask[np.arange(X.shape[0]), y.reshape(-1)] = 1
    return -(mask - y_hat).T @ X

def loss_func(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    h = softmax(X @ theta)
    return -np.log(h[np.arange(X.shape[0]), y.reshape(-1)]).sum()

def softmax_regression(data_train: np.ndarray, data_test: np.ndarray, classes:int, method = "GD", is_norm: bool = True, is_plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    x_train, y_train = np.hstack((np.ones(data_train.shape[0]).reshape(-1, 1), data_train[:, :-1])), data_train[:, -1].reshape(-1, 1).astype(int)
    x_test, y_test = np.hstack((np.ones(data_test.shape[0]).reshape(-1, 1), data_test[:, :-1])), data_test[:, -1].reshape(-1, 1).astype(int)
    
    if is_norm == True:
        x_train[:, 1:] = (x_train[:, 1:] - np.mean(x_train[:, 1:], axis=0).reshape(1, -1)) / np.std(x_train[:, 1:], axis=0).reshape(1, -1)
        x_test[:, 1:] = (x_test[:, 1:] - np.mean(x_train[:, 1:], axis=0).reshape(1, -1)) / np.std(x_train[:, 1:], axis=0).reshape(1, -1)

    if method == 'GD':
        theta = np.random.randn(x_train.shape[1], classes) # 生成一个M+1维的初始参数
        # theta = np.ones((x_train.shape[1], classes))

        loss = []
        thetas = [theta.copy()]
        for i in range(1000):
            h = softmax(x_train @ theta) # 计算预测值
            # h = np.exp(x_train @ theta) / np.sum(np.exp(x_train @ theta), axis=1).reshape(-1, 1) # 计算预测值
            # h = np.exp(x_train @ theta) / np.exp(x_train @ theta).sum(axis=1, keepdims=True) # 计算预测值
            # loss.append(-np.log(h[np.arange(x_train.shape[0]), y_train.reshape(-1)]).sum()) # 计算损失函数
            loss.append(loss_func(x_train, y_train, theta))

            # mask = np.zeros((x_train.shape[0], classes))
            # mask[np.arange(x_train.shape[0]), y_train.reshape(-1)] = 1
            # gradient = -(mask - h).T @ x_train # 计算梯度. 
            theta -= 0.0001 * g_softmax(x_train, y_train, h, classes).T # 梯度上升
            thetas.append(theta.copy())

            h_test = np.exp(x_test @ theta) / np.sum(np.exp(x_test @ theta), axis=1).reshape(-1, 1) # 计算预测值
            if np.sum(np.argmax(h_test, axis=1) != y_test.reshape(-1)) / y_test.shape[0] < 0.005: # 验证集误差小于5%时停止训练
                break
            # if i % 1000 == 0:
            #     print(f'errors: {np.sum(np.argmax(h_test, axis=1) != y_test.reshape(-1)) / y_test.shape[0]}')
        if is_plot:
            print(len(thetas), len(loss))
            print(loss[-1])
            plot_data(loss, thetas, np.hstack((x_train, y_train)), title='Softmax Regression -- GD', binary=False)
    elif method == 'L-BFGS':
        theta = np.random.randn(x_train.shape[1], classes)
        # theta = np.zeros((x_train.shape[1], classes))

        def func(theta: np.ndarray) -> float:
            theta = theta.reshape(x_train.shape[1], classes)
            return loss_func(x_train, y_train, theta)

        def gfunc(theta: np.ndarray) -> np.ndarray:
            theta = theta.reshape(x_train.shape[1], classes)
            return g_softmax(x_train, y_train, softmax(x_train @ theta), classes).T.reshape(-1, 1)
        
        thetas, loss = l_bfgs(func, gfunc, theta.reshape(-1, 1), max_iter=500)

        if is_plot:
            # print(f'the {i}th loss is {loss[-1]}')
            print(len(thetas), len(loss))
            print(loss[-1])
            for i in range(len(thetas)):
                thetas[i] = thetas[i].reshape(x_train.shape[1], classes)
            plot_data(loss, thetas, np.hstack((x_train, y_train)), title='Softmax Regression -- L-BFGS', binary=False)
    else:
        raise ValueError(f'Invalid method: {method}')

def main():
    exam_data = load_data('Exam')
    softmax_regression(exam_data[0], exam_data[1], classes=round(max(exam_data[0][:, -1]) + 1), method="GD", is_norm=True, is_plot=True)
    softmax_regression(exam_data[0], exam_data[1], classes=round(max(exam_data[0][:, -1]) + 1), method="L-BFGS", is_norm=True, is_plot=True)
    iris_data = load_data('Iris')
    softmax_regression(iris_data[0], iris_data[1], classes=round(max(iris_data[0][:, -1]) + 1), method="GD", is_norm=True, is_plot=True)
    softmax_regression(iris_data[0], iris_data[1], classes=round(max(iris_data[0][:, -1]) + 1), method="L-BFGS", is_norm=True, is_plot=True)

if __name__ == '__main__':
    main()