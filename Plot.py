import matplotlib.pyplot as plt
from Utils import *
import math
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
font = {'weight' : 'normal',
        'size'   : 21}

plt.rc('font', **font)

LI_COLORS = ['g','c','black','y','b','violet','r']
lineStyles = ['--', '-.', ':']
markers = ['8', '^', '>', 's', 'p', 'v', 'D','<', 'x']
#markers = ['s','<', '^', '>', 'v', '.']

num_colors = len(LI_COLORS)
num_styles = len(lineStyles)
num_markers = len(markers)

STD_95 = 1.96
fig_height, fig_width = 14, 8

# The method takes the data as the first parameter & name of the methods to compare & the figure name
def plot_returns(results_dir, sample_steps, compare_methods, figure_name="Return_compare.pdf",runs=1):
    indices = {}
    methods = [r[0] for r in results_dir[0]]
    #print("Methods---",methods)
    for method_index, method_name in enumerate(methods):
        #print(method_index, method_name)
        indices[method_name] = method_index
    #print(indices)
    #print(Methods.CENTROID.value in indices)
    method_names = [Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.BAYES, Methods.INCR_ADD_V]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    for method_index, method_name in enumerate(method_names):
        print(method_index, method_name)
        method_index = indices[method_name]
        if method_name.value not in compare_methods:
            continue
        method_label = method_name.value
        if method_name.value == Methods.INCR_ADD_V.value:
            method_label = "RSVF"
        elif method_name.value == Methods.BAYES.value:
            method_label = "BCI"
        elif method_name.value == Methods.HOEFFTIGHT.value:
            method_label = "Hoeffding Monotone"
        elif method_name.value == Methods.CENTROID.value:
            method_label = "Mean Transition"

        mean = np.array([r[method_index][1] for r in results_dir])

        sigma = np.array([r[method_index][5] for r in results_dir]) / np.sqrt(sample_steps)
        
        print("mean",mean, "sigma", sigma)
        
        plt.plot(sample_steps, mean, linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_label, color=LI_COLORS[method_index%num_colors])
        plt.fill_between(sample_steps, mean - STD_95 * sigma, mean + STD_95 * sigma, alpha=0.2, color=LI_COLORS[method_index%num_colors])

    plt.xlabel('Number of samples')
    plt.ylabel('Calculated return error: '+r'$\mathbb{E}[\xi]$')
    #plt.title('Expected error in return with 95% confidence interval')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.yscale('log') #, nonposy='clip'
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()
    
# The method takes the data as the first parameter & name of the methods to compare & the figure name
def plot_thresholds(results_dir, sample_steps, compare_methods, figure_name="Threshold_compare.pdf",runs=1):
    indices = {}
    methods = [r[0] for r in results_dir[0]]
    for method_index, method_name in enumerate(methods):
        indices[method_name] = method_index
        
    method_names = [Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.BAYES, Methods.INCR_ADD_V]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    for method_index, method_name in enumerate(method_names):
        method_index = indices[method_name]
        if method_name.value not in compare_methods:
            continue
        #method_label = "Robustify with Sensible Value Functions (RSVF)" if method_name.value == Methods.INCR_ADD_V.value else method_name.value
        method_label = method_name.value
        if method_name.value == Methods.INCR_ADD_V.value:
            method_label = "RSVF"
        elif method_name.value == Methods.BAYES.value:
            method_label = "BCI"
        elif method_name.value == Methods.HOEFFTIGHT.value:
            method_label = "Hoeffding Monotone"
        elif method_name.value == Methods.CENTROID.value:
            method_label = "Mean Transition"
        mean = np.array([r[method_index][2] for r in results_dir])
        sigma = np.array([r[method_index][6] for r in results_dir]) / np.sqrt(len(sample_steps))
        plt.plot(sample_steps, [r[method_index][2] for r in results_dir], linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_label, color=LI_COLORS[method_index%num_colors])
        plt.fill_between(sample_steps, mean - STD_95 * sigma, mean + STD_95 * sigma, alpha=0.5, color=LI_COLORS[method_index%num_colors])
    
    plt.xlabel('Number of samples')
    plt.ylabel(r'$L_1$'+' threshold')
    #plt.title('Size of '+r'$L_1$'+' ball with 95% confidence interval')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()

# The method takes the data as the first parameter & name of the methods to compare & the figure name
def plot_violations(results_dir, sample_steps, compare_methods, figure_name="Violations_compare.pdf"):
    indices = {}
    methods = [r[0] for r in results_dir[0]]
    for method_index, method_name in enumerate(methods):
        indices[method_name] = method_index
        
    method_names = [Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.BAYES, Methods.INCR_ADD_V]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')

    for method_index, method_name in enumerate(method_names):
        method_index = indices[method_name]
        if method_name.value not in compare_methods:
            continue
        #method_label = "Robustify with Sensible Value Functions (RSVF)" if method_name.value == Methods.INCR_ADD_V.value else method_name.value
        method_label = method_name.value
        if method_name.value == Methods.INCR_ADD_V.value:
            method_label = "RSVF"
        elif method_name.value == Methods.BAYES.value:
            method_label = "BCI"
        elif method_name.value == Methods.HOEFFTIGHT.value:
            method_label = "Hoeffding Monotone"
        elif method_name.value == Methods.CENTROID.value:
            method_label = "Mean Transition"
        plt.plot(sample_steps, [r[method_index][3] for r in results_dir], linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_label, color=LI_COLORS[method_index%num_colors])
    plt.xlabel('Number of samples')
    plt.ylabel('Fraction violated')
    #plt.title('L1 threshold values')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()

compare_methods = [Methods.BAYES.value, Methods.CENTROID.value, Methods.HOEFF.value, Methods.HOEFFTIGHT.value,  Methods.INCR_ADD_V.value]
# The generic method to plot data. First param is the x axis, second is a list of y axis data
#plot_MDP_returns
def plot_MDP_returns(X, Data, figure_name="Generic_Plot.pdf"):
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    indices = {}
    methods = [Methods.BAYES, Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.EM, Methods.INCR_REPLACE_V, Methods.INCR_ADD_V ]
    for method_index, method_name in enumerate(methods):
        indices[method_name] = method_index
        
    method_names = [Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.BAYES, Methods.INCR_ADD_V]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    for method_index, method_name in enumerate(method_names):
        method_index = indices[method_name]
        if method_name.value not in compare_methods:
            continue
        method_label = method_name.value
        if method_name.value == Methods.INCR_ADD_V.value:
            method_label = "RSVF"
        elif method_name.value == Methods.BAYES.value:
            method_label = "BCI"
        elif method_name.value == Methods.HOEFFTIGHT.value:
            method_label = "Hoeffding Monotone"
        elif method_name.value == Methods.CENTROID.value:
            method_label = "Mean Transition"
        Y = np.array([d[0] for d in Data[method_index]])
        sigma = np.array([d[1] for d in Data[method_index]]) / np.sqrt(X)
        plt.plot(X, Y, linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_label, color=LI_COLORS[method_index%num_colors])
        plt.fill_between(X, Y - STD_95 * sigma, Y + STD_95 * sigma, alpha=0.2, color=LI_COLORS[method_index%num_colors])
        
    plt.xlabel('Number of samples')
    plt.ylabel('Calculated return error: '+r'$\mathbb{E}[\xi]$')
    #plt.title(plot_title)
    #plt.legend(loc=legend_pos)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()


def plot_MDP_violations(X, Data, figure_name="Generic_Plot.pdf"):
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    indices = {}
    methods = [Methods.BAYES, Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.EM, Methods.INCR_REPLACE_V, Methods.INCR_ADD_V ]
    for method_index, method_name in enumerate(methods):
        indices[method_name] = method_index
        
    method_names = [Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.BAYES, Methods.INCR_ADD_V]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    for method_index, method_name in enumerate(method_names):
        method_index = indices[method_name]
        if method_name.value not in compare_methods:
            continue
        method_label = method_name.value
        if method_name.value == Methods.INCR_ADD_V.value:
            method_label = "RSVF"
        elif method_name.value == Methods.BAYES.value:
            method_label = "BCI"
        elif method_name.value == Methods.HOEFFTIGHT.value:
            method_label = "Hoeffding Monotone"
        elif method_name.value == Methods.CENTROID.value:
            method_label = "Mean Transition"
        Y = np.array([d[0] for d in Data[method_index]])
        #sigma = np.array([d[1] for d in Data[method_index]]) / np.sqrt(X)
        plt.plot(X, Y, linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_label, color=LI_COLORS[method_index%num_colors])
        #plt.fill_between(X, Y - STD_95 * sigma, Y + STD_95 * sigma, alpha=0.2, color=LI_COLORS[method_index%num_colors])
        
    plt.xlabel('Number of samples')
    plt.ylabel('Fraction violated')
    #plt.title(plot_title)
    #plt.legend(loc=legend_pos)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()

# A generic simple plot
def simple_generic_plot(X, Y, x_lab="x axis", y_lab="y axis", legend_pos="upper right", plot_title = "", figure_name="Simple_Generic_Plot.pdf"):
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(X, Y, '+-', label = "Value Function")

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(plot_title)
    plt.legend(loc=legend_pos)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()

    