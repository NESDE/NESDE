from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_quantiles(X, Q=100, assume_sorted=False):
    if not assume_sorted:
        X = sorted(X)
    N = len(X)
    try:
        n = len(Q)
        Q = np.array(Q)
    except:
        Q = np.arange(0,1+1e-6,1/Q)
        Q[-1] = 1
        n = len(Q)
    x = [X[int(q*(N-1))] for q in Q]
    return x, Q

def plot_quantiles(x, ax=None, Q=100, showmeans=True, means_args=None, **kwargs):
    if ax is None: ax = Axes(1,1)[0]
    m = np.mean(x)
    x, q = get_quantiles(x, Q)
    h = ax.plot(100*q, x, '-', **kwargs)
    if showmeans:
        if means_args is None: means_args = {}
        ax.axhline(m, linestyle='--', color=h[0].get_color(), **means_args)
    return ax

def labels(ax, xlab=None, ylab=None, title=None, fontsize=12):
    if isinstance(fontsize, int):
        fontsize = 3*[fontsize]
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=fontsize[0])
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=fontsize[1])
    if title is not None:
        ax.set_title(title, fontsize=fontsize[2])

def fontsize(ax, labs=16, ticks=12, leg=None, draw=True, wait=0):
    if wait:
        sleep(wait)
    if draw:
        plt.draw()
    if labs is not None:
        if not isinstance(labs, (tuple,list)):
            labs = 3*[labs]
            ax.set_xlabel(ax.get_xlabel(), fontsize=labs[0])
            ax.set_ylabel(ax.get_ylabel(), fontsize=labs[1])
            ax.set_title(ax.get_title(), fontsize=labs[2])
    if ticks is not None:
        if not isinstance(ticks, (tuple,list)):
            ticks = 2*[ticks]
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=ticks[0])
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=ticks[1])
    if leg is not None:
        ax.legend(fontsize=leg)

class Axes:
    def __init__(self, N, W=2, axsize=(5,3.5), grid=1, fontsize=13):
        self.fontsize = fontsize
        self.N = N
        self.W = W
        self.H = int(np.ceil(N/W))
        self.axs = plt.subplots(self.H, self.W, figsize=(self.W*axsize[0], self.H*axsize[1]))[1]
        for i in range(self.N):
            if grid == 1:
                self[i].grid(color='k', linestyle=':', linewidth=0.3)
            elif grid ==2:
                self[i].grid()
        for i in range(self.N, self.W*self.H):
            self[i].axis('off')

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if self.H == 1 and self.W == 1:
            return self.axs
        elif self.H == 1 or self.W == 1:
            return self.axs[item]
        return self.axs[item//self.W, item % self.W]

    def labs(self, item, *args, **kwargs):
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.fontsize
        labels(self[item], *args, **kwargs)
