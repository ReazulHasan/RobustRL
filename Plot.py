import matplotlib.pyplot as plt
import math
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
font = {'weight' : 'normal',
        'size'   : 21}

plt.rc('font', **font)

LI_COLORS = ['black','c','y','g','b','violet','r']
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
    method_names = [r[0] for r in results_dir[0]]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    for method_index, method_name in enumerate(method_names):
        if method_name not in compare_methods:
            continue
        mean = np.array([r[method_index][1] for r in results_dir])
        #print("np.sqrt(sample_steps)",np.sqrt(sample_steps))
        sigma = np.array([r[method_index][5] for r in results_dir]) / np.sqrt(sample_steps)
        #print(mean, sigma, mean + 1.96 * sigma)
        plt.plot(sample_steps, mean, linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_name.value, color=LI_COLORS[method_index%num_colors])
        plt.fill_between(sample_steps, mean - STD_95 * sigma, mean + STD_95 * sigma, alpha=0.2, color=LI_COLORS[method_index%num_colors])

    plt.xlabel('Number of samples'+r'$(N)$')
    plt.ylabel('Calculated return error: '+r'$\rho^* - \rho(\xi)$')
    #plt.title('Expected error in return with 95% confidence interval')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.yscale('log') #, nonposy='clip'
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()
    
# The method takes the data as the first parameter & name of the methods to compare & the figure name
def plot_thresholds(results_dir, sample_steps, compare_methods, figure_name="Threshold_compare.pdf",runs=1):
    method_names = [r[0] for r in results_dir[0]]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    for method_index, method_name in enumerate(method_names):
        if method_name not in compare_methods:
            continue
        mean = np.array([r[method_index][2] for r in results_dir])
        sigma = np.array([r[method_index][6] for r in results_dir]) / np.sqrt(len(sample_steps))
        plt.plot(sample_steps, [r[method_index][2] for r in results_dir], linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_name.value, color=LI_COLORS[method_index%num_colors])
        plt.fill_between(sample_steps, mean - STD_95 * sigma, mean + STD_95 * sigma, alpha=0.5, color=LI_COLORS[method_index%num_colors])
    
    plt.xlabel('Number of samples'+r'$(N)$')
    plt.ylabel(r'$L_1$'+' threshold')
    #plt.title('Size of '+r'$L_1$'+' ball with 95% confidence interval')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()

# The method takes the data as the first parameter & name of the methods to compare & the figure name
def plot_violations(results_dir, sample_steps, compare_methods, figure_name="Violations_compare.pdf"):
    method_names = [r[0] for r in results_dir[0]]
    # violations
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    #plt.subplot(121)
    for method_index, method_name in enumerate(method_names):
        if method_name not in compare_methods:
            continue
        plt.plot(sample_steps, [r[method_index][3] for r in results_dir], linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_name.value, color=LI_COLORS[method_index%num_colors])
    plt.xlabel('Number of samples')
    plt.ylabel('Fraction violated')
    #plt.title('L1 threshold values')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()
    
# The generic method to plot data. First param is the x axis, second is a list of y axis data
def generic_plot(X, Data, x_lab="x axis", y_lab="y axis", legend_pos="lower right", plot_title = "", figure_name="Generic_Plot.pdf"):
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    for method_index in range(Methods.NUM_METHODS.value):
        if LI_METHODS[method_index] is Methods.EM or LI_METHODS[method_index] is Methods.INCR_REPLACE_V:
            continue
        plt.plot(X, Data[method_index], linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = LI_METHODS[method_index].value)

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(plot_title)
    plt.legend(loc=legend_pos)
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

    