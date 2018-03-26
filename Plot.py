import matplotlib.pyplot as plt
import math
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
font = {'weight' : 'normal',
        'size'   : 19}

plt.rc('font', **font)

LI_COLORS = ['b','c','r','g','m']
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
        sigma = np.array([r[method_index][5] for r in results_dir])
        #print(mean, sigma, mean + 1.96 * sigma)
        plt.plot(sample_steps, mean, '+-', label = method_name.value, color=LI_COLORS[method_index])
        plt.fill_between(sample_steps, mean - STD_95 * sigma / math.sqrt(runs), mean + STD_95 * sigma / math.sqrt(runs), alpha=0.2, color=LI_COLORS[method_index])

    plt.xlabel('Number of samples'+r'$(N)$')
    plt.ylabel('Calculated return error: '+r'$\rho^* - \rho(\xi)$')
    #plt.title('Expected error in return with 95% confidence interval')
    plt.legend(loc='upper right')
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
        sigma = np.array([r[method_index][6] for r in results_dir])
        plt.plot(sample_steps, [r[method_index][2] for r in results_dir], '+-', label = method_name.value, color=LI_COLORS[method_index])
        plt.fill_between(sample_steps, mean - STD_95 * sigma / math.sqrt(runs), mean + STD_95 * sigma / math.sqrt(runs), alpha=0.3, color=LI_COLORS[method_index])
    
    plt.xlabel('number of samples'+r'$(N)$')
    plt.ylabel(r'$L_1$'+' threshold')
    #plt.title('Size of '+r'$L_1$'+' ball with 95% confidence interval')
    plt.legend(loc='upper right')
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
        plt.plot(sample_steps, [r[method_index][3] for r in results_dir], '+-', label = method_name.value, color=LI_COLORS[method_index])
    plt.xlabel('number of samples')
    plt.ylabel('fraction violated')
    #plt.title('L1 threshold values')
    plt.legend(loc='upper right')
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()
    
# The generic method to plot data. First param is the x axis, second is a list of y axis data
def generic_plot(X, Data, x_lab="x axis", y_lab="y axis", legend_pos="upper right", plot_title = "", figure_name="Generic_Plot.pdf"):
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    for index in range(Methods.NUM_METHODS.value):
        plt.plot(X, Data[index], '+-', label = LI_METHODS[index].value)

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(plot_title)
    plt.legend(loc=legend_pos)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name+".pdf")
    plt.show()
        
# A generic simple plot
def simple_generic_plot(X, Y, x_lab="x axis", y_lab="y axis", legend_pos="upper right", plot_title = "", figure_name="Generic_Plot.pdf"):
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(X, Y, '+-', label = "Value Function")

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(plot_title)
    plt.legend(loc=legend_pos)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name+".pdf")
    plt.show()

    