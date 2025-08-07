import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from gen_data import get_train_two_feats_gau, get_test_two_feats_gau

plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

palette = sns.color_palette("muted", n_colors=2)
sns.set_palette(palette)

bound_palette = [sns.dark_palette("#F98", reverse=False, n_colors=1)[0], sns.dark_palette("#F98", reverse=False, n_colors=1)[0]]

def set_ax_style(ax):
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linestyle('-')
    ax.spines['bottom'].set_linestyle('-')
    ax.spines['right'].set_linestyle('-')
    ax.spines['top'].set_linestyle('-')
    return ax

def plot_decision_boundary(decision_boundary, ax, method, x_range, y_range, color='black', linestyle='-', linewidth=2, alpha=0.8):
    """Plot decision boundary for a given method"""
    weight = decision_boundary[method]['weight']
    bias = decision_boundary[method]['bias']

    x_range = [-4, 9]
    y_range = [-4, 9]
    
    # Decision boundary: w1*x + w2*y + b = 0
    w1, w2 = weight
    b = bias  # Remove the [0] indexing since bias is a scalar
    
    # Check if w2 is too close to zero
    if abs(w2) < 1e-2:
        # If w2 ≈ 0, then w1*x + b = 0, so x = -b/w1
        x_val = -b/w1
        if x_range[0] <= x_val <= x_range[1]:
            y_vals = np.linspace(y_range[0], y_range[1], 100)
            x_vals = np.full_like(y_vals, x_val)
            ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, 
                    label=f'{method.upper()}')
    else:
        # Normal case: y = -(w1*x + b) / w2
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        y_vals = -(w1 * x_vals + b) / w2
        
        # Filter points within y_range
        mask = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]
        
        ax.plot(x_plot, y_plot, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, 
                label=f'{method.upper()}')

# set random seed
seed = 0
do_balance = True

def plot_bound(decision_boundary, do_balance, seed=0):
    # Generate data
    if do_balance:
        train_x, train_y = get_train_two_feats_gau(500, 0.1, seed=seed)
    else:
        train_x, train_y = get_train_two_feats_gau(500, 0.5, seed=seed)

    test_x, test_y = get_test_two_feats_gau(500, 0.5, seed=seed)

    test_x_2, test_y_2 = get_test_two_feats_gau(500, 0.5, True, seed=seed)

    train_x = train_x + 1e-7*np.random.randn(*train_x.shape)
    test_x = test_x + 1e-7*np.random.randn(*test_x.shape)
    test_x_2 = test_x_2 + 1e-7*np.random.randn(*test_x_2.shape)

    # Determine plot ranges
    x_range = [-5, 10]
    y_range = [-5, 10]

    # Create 2D density plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Training data - both classes
    sns.kdeplot(x=train_x[train_y==0, 0], y=train_x[train_y==0, 1], 
            cmap='Blues', fill=False, alpha=0.8, ax=axes[0], levels=10)
    sns.kdeplot(x=train_x[train_y==1, 0], y=train_x[train_y==1, 1], 
            cmap='Reds', fill=False, alpha=0.8, ax=axes[0], levels=10)
    # Add decision boundaries
    plot_decision_boundary(decision_boundary, axes[0], 'bl', x_range, y_range, color=bound_palette[0], linestyle='-', linewidth=1.8, alpha=0.9)
    plot_decision_boundary(decision_boundary, axes[0], 'cls_unbias', x_range, y_range, color=bound_palette[1], linestyle='--', linewidth=1.8, alpha=0.9)
    # axes[0].set_title('Training Data')
    axes[0].set_xlabel('$f_1$')
    axes[0].set_ylabel('$f_2$')
    axes[0].set_xlim(x_range)
    axes[0].set_ylim(y_range)
    # Add custom legend with line elements
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=palette[0], linewidth=2, label='neg ($y=0$)'),
                    Line2D([0], [0], color=palette[1], linewidth=2, label='pos ($y=1$)'),
                    Line2D([0], [0], color=bound_palette[0], linewidth=1.8, label='ERM'),
                    Line2D([0], [0], color=bound_palette[1], linewidth=1.8, linestyle='--', label='Cls-unbias')]
    axes[0].legend(handles=legend_elements, loc='upper right', frameon=True,)
    set_ax_style(axes[0])

    # Test data 1 - both classes
    sns.kdeplot(x=test_x[test_y==0, 0], y=test_x[test_y==0, 1], 
                cmap='Blues', fill=False, alpha=0.8, ax=axes[1])
    sns.kdeplot(x=test_x[test_y==1, 0], y=test_x[test_y==1, 1], 
                cmap='Reds', fill=False, alpha=0.8, ax=axes[1])
    # Add decision boundaries
    plot_decision_boundary(decision_boundary, axes[1], 'bl', x_range, y_range, color=bound_palette[0], linestyle='-', linewidth=1.8, alpha=0.9)
    plot_decision_boundary(decision_boundary, axes[1], 'cls_unbias', x_range, y_range, color=bound_palette[1], linestyle='--', linewidth=1.8, alpha=0.9)
    # axes[1].set_title('Test Data 1')
    axes[1].set_xlabel('$f_1$')
    axes[1].set_ylabel('$f_2$')
    axes[1].set_xlim(x_range)
    axes[1].set_ylim(y_range)
    axes[1].legend(handles=legend_elements, loc='upper right', frameon=True)
    set_ax_style(axes[1])

    # Test data 2 - both classes
    sns.kdeplot(x=test_x_2[test_y_2==0, 0], y=test_x_2[test_y_2==0, 1], 
                cmap='Blues', fill=False, alpha=0.8, ax=axes[2])
    sns.kdeplot(x=test_x_2[test_y_2==1, 0], y=test_x_2[test_y_2==1, 1], 
                cmap='Reds', fill=False, alpha=0.8, ax=axes[2])
    # Add decision boundaries
    plot_decision_boundary(decision_boundary, axes[2], 'bl', x_range, y_range, color=bound_palette[0], linestyle='-', linewidth=1.8, alpha=0.9)
    plot_decision_boundary(decision_boundary, axes[2], 'cls_unbias', x_range, y_range, color=bound_palette[1], linestyle='--', linewidth=1.8, alpha=0.9)
    # axes[2].set_title('Test Data 2')
    axes[2].set_xlabel('$f_1$')
    axes[2].set_ylabel('$f_2$')
    axes[2].set_xlim(x_range)
    axes[2].set_ylim(y_range)
    axes[2].legend(handles=legend_elements, loc='upper right', frameon=True)
    set_ax_style(axes[2])

    plt.tight_layout()
    if do_balance:
        f_name = 'balanced'
    else:
        f_name = 'imbalanced'
    os.makedirs(f'figs/{f_name}', exist_ok=True)
    plt.savefig(f'figs/{f_name}/2d_density.pdf', bbox_inches='tight')
    plt.show()

    # Save individual subplots
    # Training data
    fig_train, ax_train = plt.subplots(1, 1, figsize=(4.7, 4.7))
    sns.kdeplot(x=train_x[train_y==0, 0], y=train_x[train_y==0, 1], 
                cmap='Blues', fill=False, alpha=0.8, ax=ax_train, levels=10)
    sns.kdeplot(x=train_x[train_y==1, 0], y=train_x[train_y==1, 1], 
                cmap='Reds', fill=False, alpha=0.8, ax=ax_train, levels=10)
    # Add decision boundaries
    plot_decision_boundary(decision_boundary, ax_train, 'bl', x_range, y_range, color=bound_palette[0], linestyle='-', linewidth=1.8, alpha=0.9)
    plot_decision_boundary(decision_boundary, ax_train, 'cls_unbias', x_range, y_range, color=bound_palette[1], linestyle='--', linewidth=1.8, alpha=0.9)
    # ax_train.set_title('Training Data')
    ax_train.set_xlabel('$f_1$')
    ax_train.set_ylabel('$f_2$')
    ax_train.set_xlim(x_range)
    ax_train.set_ylim(y_range)
    ax_train.legend(handles=legend_elements, loc='upper right', frameon=True)
    set_ax_style(ax_train)
    plt.tight_layout()
    plt.savefig(f'figs/{f_name}/train_density.pdf', bbox_inches='tight')
    plt.close()

    # Test data 1
    fig_test1, ax_test1 = plt.subplots(1, 1, figsize=(4.7, 4.7))
    sns.kdeplot(x=test_x[test_y==0, 0], y=test_x[test_y==0, 1], 
                cmap='Blues', fill=False, alpha=0.8, ax=ax_test1)
    sns.kdeplot(x=test_x[test_y==1, 0], y=test_x[test_y==1, 1], 
                cmap='Reds', fill=False, alpha=0.8, ax=ax_test1)
    # Add decision boundaries
    plot_decision_boundary(decision_boundary, ax_test1, 'bl', x_range, y_range, color=bound_palette[0], linestyle='-', linewidth=1.8, alpha=0.9)
    plot_decision_boundary(decision_boundary, ax_test1, 'cls_unbias', x_range, y_range, color=bound_palette[1], linestyle='--', linewidth=1.8, alpha=0.9)
    # ax_test1.set_title('Test Data 1')
    ax_test1.set_xlabel('$f_1$')
    ax_test1.set_ylabel('$f_2$')
    ax_test1.set_xlim(x_range)
    ax_test1.set_ylim(y_range)
    ax_test1.legend(handles=legend_elements, loc='upper right', frameon=True)
    set_ax_style(ax_test1)
    plt.tight_layout()
    plt.savefig(f'figs/{f_name}/test1_density.pdf', bbox_inches='tight')
    plt.close()

    # Test data 2
    fig_test2, ax_test2 = plt.subplots(1, 1, figsize=(4.7, 4.7))
    sns.kdeplot(x=test_x_2[test_y_2==0, 0], y=test_x_2[test_y_2==0, 1], 
                cmap='Blues', fill=False, alpha=0.8, ax=ax_test2)
    sns.kdeplot(x=test_x_2[test_y_2==1, 0], y=test_x_2[test_y_2==1, 1], 
                cmap='Reds', fill=False, alpha=0.8, ax=ax_test2)
    # Add decision boundaries
    plot_decision_boundary(decision_boundary, ax_test2, 'bl', x_range, y_range, color=bound_palette[0], linestyle='-', linewidth=1.8, alpha=0.9)
    plot_decision_boundary(decision_boundary, ax_test2, 'cls_unbias', x_range, y_range, color=bound_palette[1], linestyle='--', linewidth=1.8, alpha=0.9)
    # ax_test2.set_title('Test Data 2')
    ax_test2.set_xlabel('$f_1$')
    ax_test2.set_ylabel('$f_2$')
    ax_test2.set_xlim(x_range)
    ax_test2.set_ylim(y_range)
    ax_test2.legend(handles=legend_elements, loc='upper right', frameon=True)
    set_ax_style(ax_test2)
    plt.tight_layout()
    plt.savefig(f'figs/{f_name}/test2_density.pdf', bbox_inches='tight')
    plt.close()

    print("Generated 2D density plots with decision boundaries for the training data, test data 1, and test data 2.")
    print("Files saved as:")
    print("- Combined plot: figs/{f_name}/2d_density.pdf")
    print("- Individual plots: figs/{f_name}/train_density.pdf, figs/{f_name}/test1_density.pdf, figs/{f_name}/test2_density.pdf")