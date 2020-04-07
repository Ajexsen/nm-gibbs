import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns


def prep_group(grouped_result):
    return [list(grouped_result[i]) for i in range(len(grouped_result))]


def plot_all(group_features, path_img='img/out', save_img=False):
    for i in range(len(group_features)):
        plot_features(group_features[i], i, path_img, save_img)


def all_match(group_features, n):
    for i in range(len(group_features)):
        if len(group_features[i]) >= n:
            plot_features(group_features[i], i)
        else:
            continue


def plot_features(group_feature, index, path_img='img/out', save_img=False):
    kn = len(group_feature)
    ki = [group_feature[i][:784].reshape((28, 28)) for i in range(kn)]
    # ki = [group_feature[index][i,:784].reshape((28,28)) for i in range(n)]
    # ki = [group_feature[index] for i in range(n)]

    plt.gray()
    plt.tight_layout()
    plt.axis('off')

    sq = np.ceil(np.sqrt(kn)).astype(int)

    if kn <= 3:
        # f, axarr = plt.subplots(2, 2)
        #        plt.imshow(ki[2])
        # plt.set_title("local")
        f, axarr = plt.subplots(1, 1)
        axarr.imshow(ki[0])
        axarr.axis('off')
        # plt.axis('off')
    else:
        f, axarr = plt.subplots(sq, sq)
        # axarr[0, 0].imshow(ki[0])
        # axarr[0, 0].set_title("Mean")
        # axarr[0, 0].axis('off')

        axarr[0, 0].imshow(ki[0])
        axarr[0, 0].set_title("Global")
        axarr[0, 0].axis('off')

        for i in range(sq ** 2):
            r = int(i / sq)
            c = i % sq
            axarr[r, c].axis('off')
            if i in range(1, kn):
                axarr[r, c].imshow(ki[i])
                axarr[r, c].set_title("local" + str(i - 1))
    if save_img:
        f.savefig(path_img + str(index) + '.png', dpi=100)


def plot_curve(log_track, k_track, log_var=None, k_var=None, saveFile=False, show_label=False, fileName="plot", y_int=True, c1_lh=None, c2_lh=None):
    assert len(log_track) == len(k_track)
    n = len(log_track)

    index = [i for i in range(1, n + 1)]
 
    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot2grid((2,1),(1,0))    
    ax2 = plt.subplot2grid((2,1),(0,0))#,rowspan = 2)

    color_gen = 'black'
    color_log = 'xkcd:charcoal grey'
    color_k = 'orange'
    color_error = 'lightgrey'
    color_eline = 'white'
    loglike = 'log-likelihood'
    k = 'k'
    er = 'error'
    
    ax1.set_xlabel('iteration')
    
    ax1.set_ylabel(k, color=color_gen)
    ax1.tick_params(axis='y', labelcolor=color_gen)
    ax1.grid(False)
    if y_int:
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
 
    ax2.set_ylabel(loglike, color=color_gen)
    ax2.tick_params(axis='y', labelcolor=color_gen)
    ax2.get_xaxis().set_visible(False)
    ax2.grid(False)
    
    lineWidth = 3
    tran = 0.7
    markSize = 5
    elineWidth = 0.5

    if log_var is None:
        ax2.plot(index, log_track, color=color_log, linewidth=lineWidth, alpha=tran, label=loglike,
                 marker='o', markersize=markSize)
    else:
        ax2.errorbar(index, log_track, log_var, fmt='-o', color=color_log, linewidth=lineWidth,
                     ecolor=color_eline, elinewidth=elineWidth, alpha=tran, markersize=markSize, label=loglike)
        ax2.fill_between(index, log_track-log_var, log_track+log_var, alpha=tran, color=color_error, label=er)
    
    if c1_lh is not None:
        ax2.set_ylim(c1_lh)
    
    if k_var is None:
        ax1.plot(index, k_track, color=color_k, linewidth=lineWidth, alpha=tran, label=k,
                 marker='o', markersize=markSize)
    else:
        ax1.errorbar(index, k_track, k_var, fmt='-o', color=color_k, linewidth=lineWidth,
                     ecolor=color_eline, elinewidth=tran, alpha=tran, markersize=markSize, label=k)
        ax1.fill_between(index, k_track-k_var, k_track+k_var, alpha=tran, color=color_error)
        
    if c2_lh is not None:
        ax1.set_ylim(c2_lh)
    
    if show_label is True:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles2+handles1, labels2+labels1, loc="lower right")
        
    plt.tight_layout()    
    if saveFile:
        plt.savefig(fileName, dpi=400)
    plt.show()
    
    

def plot_curve2(log_track, k_track, log_var=None, k_var=None, saveFile=False, show_label=False, fileName="plot", y_int=True, c1_lh=None, c2_lh=None):
    assert len(log_track) == len(k_track)
    n = len(log_track)

    index = [i for i in range(1, n + 1)]
 
    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot2grid((2,1),(1,0))    
    ax2 = plt.subplot2grid((2,1),(0,0))#,rowspan = 2)

    color_gen = 'black'
    color_log = 'xkcd:charcoal grey'
    color_k = 'xkcd:scarlet'#'red'
    color_error = 'lightgrey'
    color_eline = 'white'
    loglike = 'log-likelihood'
    k = 'accuracy'
    er = 'error'
    
    ax1.set_xlabel('iteration')
    
    ax1.set_ylabel(k, color=color_gen)
    ax1.tick_params(axis='y', labelcolor=color_gen)
    ax1.grid(False)
    if y_int:
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
 
    ax2.set_ylabel(loglike, color=color_gen)
    ax2.tick_params(axis='y', labelcolor=color_gen)
    ax2.get_xaxis().set_visible(False)
    ax2.grid(False)
    
    lineWidth = 3
    tran = 0.7
    markSize = 5
    elineWidth = 0.5

    if log_var is None:
        ax2.plot(index, log_track, color=color_log, linewidth=lineWidth, alpha=tran, label=loglike,
                 marker='o', markersize=markSize)
    else:
        ax2.errorbar(index, log_track, log_var, fmt='-o', color=color_log, linewidth=lineWidth,
                     ecolor=color_eline, elinewidth=elineWidth, alpha=tran, markersize=markSize, label=loglike)
        ax2.fill_between(index, log_track-log_var, log_track+log_var, alpha=tran, color=color_error, label=er)

    if c1_lh is not None:
        ax2.set_ylim(c1_lh)
    
    if k_var is None:
        ax1.plot(index, k_track, color=color_k, linewidth=lineWidth, alpha=tran, label=k,
                 marker='o', markersize=markSize)
    else:
        ax1.errorbar(index, k_track, k_var, fmt='-o', color=color_k, linewidth=lineWidth,
                     ecolor=color_eline, elinewidth=tran, alpha=tran, markersize=markSize, label=k)
        ax1.fill_between(index, k_track-k_var, k_track+k_var, alpha=tran, color=color_error)
        
    if c2_lh is not None:
        ax1.set_ylim(c2_lh)

    if show_label is True:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles2+handles1, labels2+labels1, loc="lower right")
    
    plt.tight_layout()    
    if saveFile:
        plt.savefig(fileName, dpi=400)
    plt.show()