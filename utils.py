import matplotlib.collections as mcoll
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

def _add_grid_lines(ca, ax, show_grid):
    """
    Adds grid lines to the plot.

    :param ca: the 2D cellular automaton to plot

    :param ax: the Matplotlib axis object

    :param show_grid: whether to display the grid lines

    :return: the grid object
    """
    grid_linewidth = 0.0
    if show_grid:
        plt.xticks(np.arange(-.5, len(ca[0][0]), 1), "")
        plt.yticks(np.arange(-.5, len(ca[0]), 1), "")
        plt.tick_params(axis='both', which='both', length=0)
        grid_linewidth = 0.5
    vertical = np.arange(-.5, len(ca[0][0]), 1)
    horizontal = np.arange(-.5, len(ca[0]), 1)
    lines = ([[(x, y) for y in (-.5, horizontal[-1])] for x in vertical] +
             [[(x, y) for x in (-.5, vertical[-1])] for y in horizontal])
    grid = mcoll.LineCollection(lines, linestyles='-', linewidths=grid_linewidth, color='grey')
    ax.add_collection(grid)

    return grid

def custom_plot2d(ca, timestep=None, title='', *, colormap='Greys', show_grid=False, show_margin=True, figsize=(6, 6),
                  scale=0.6, show=False,vmin=0,vmax=2, **imshow_kwargs):
    """
    2D 可视化函数，支持设置图像大小,支持多色映射。
    """
    cmap = plt.get_cmap(colormap)
    fig, ax = plt.subplots(figsize=figsize)  # 使用 figsize 控制图像大小
    plt.title(title)
    if not show_margin:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if timestep is not None:
        data = ca[timestep]
    else:
        data = ca[-1]

    _add_grid_lines(ca, ax, show_grid)

    im = ax.imshow(data, interpolation='none', cmap=cmap, **imshow_kwargs,vmin=vmin, vmax=vmax)
    if not show_margin:
        baseheight, basewidth = im.get_size()
        fig.set_size_inches(basewidth * scale, baseheight * scale, forward=True)
    if show:
        plt.show()

def custom_plot2d_animate(ca, title='', *, colormap='Greys', show_grid=False, show_margin=True, scale=0.6, dpi=80,
                   interval=50, save=False, autoscale=False, show=False,figsize=(10, 10),vmin=0,vmax=2,
                   **imshow_kwargs):
    '''
    2D 动画可视化函数，支持设置图像大小,支持多色映射。
    '''
    cmap = plt.get_cmap(colormap)
    fig, ax = plt.subplots(figsize=figsize)  # Set the figure size
    plt.title(title)

    if not show_margin:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    grid = _add_grid_lines(ca, ax, show_grid)

    # Use vmin and vmax for imshow
    im = plt.imshow(ca[0], animated=True, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
    if not show_margin:
        baseheight, basewidth = im.get_size()
        fig.set_size_inches(basewidth * scale, baseheight * scale, forward=True)

    # Update function for the animation
    i = {'index': 0}

    def updatefig(*args):
        i['index'] += 1
        if i['index'] == len(ca):
            i['index'] = 0
        im.set_array(ca[i['index']])
        if autoscale:
            im.autoscale()  # Autoscale if enabled
        return im, grid

    ani = animation.FuncAnimation(fig, updatefig, interval=interval, blit=True, save_count=len(ca))

    if save:
        ani.save('evolved.gif', dpi=dpi, writer="imagemagick")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani

def generate_exposure_infection_maps(num_rows, num_cols, exposure_mean=3, infection_mean=5, exposure_std=1.0, infection_std=1.0, device='cuda:0'):
    """
    生成潜伏期和发作期的映射（均为整数）
    
    """

    # 生成潜伏期（正态分布，均值为3）
    exposure_map = np.random.normal(exposure_mean, exposure_std, (num_rows, num_cols))
    exposure_map = np.round(exposure_map).astype(int)  
    exposure_map = torch.tensor(exposure_map, dtype=torch.long, device=device)
    
    # 生成发作期（正态分布，均值为5）
    infection_map = np.random.normal(infection_mean, infection_std, (num_rows, num_cols))
    infection_map = np.round(infection_map).astype(int)  
    infection_map = torch.tensor(infection_map, dtype=torch.long, device=device)
    
    return infection_map,exposure_map



def plot_smooth_data(d_account=None, a_account=None, window_size=10, 
                          left_title='Death counts over time', left_label='Death counts',
                          right_title='Exposed counts over time', right_label='Exposed counts'):
    """
    绘制左右两张平滑曲线图：
    左图显示 d_account 右图显示 a_account。
    """
    # 平滑曲线计算
    smoothed_d_account = np.convolve(d_account, np.ones(window_size) / window_size, mode='valid')
    smoothed_a_account = np.convolve(a_account, np.ones(window_size) / window_size, mode='valid')
    
    # 时间步范围
    time_steps_d = range(len(d_account))
    time_steps_a = range(len(a_account))
    
    # 创建绘图窗口
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1行2列子图
    
    # 左图：d_account
    axes[0].plot(time_steps_d, d_account, label=left_label, color='red', alpha=0.5)
    axes[0].plot(time_steps_d[window_size - 1:], smoothed_d_account, label='Smoothed', color='blue', linewidth=2)
    axes[0].set_title(left_title, fontsize=14)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel(left_label, fontsize=12)
    axes[0].legend()
    axes[0].grid(True)
    
    # 右图：a_account
    axes[1].plot(time_steps_a, a_account, label=right_label, color='green', alpha=0.5)
    axes[1].plot(time_steps_a[window_size - 1:], smoothed_a_account, label='Smoothed', color='purple', linewidth=2)
    axes[1].set_title(right_title, fontsize=14)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel(right_label, fontsize=12)
    axes[1].legend()
    axes[1].grid(True)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

