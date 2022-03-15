
import matplotlib.pyplot as plt
import os
def visualize( X, Y, color, new_plot=False, figure=None):
    if new_plot:
        fig = plt.figure()
    else:
        fig = figure
    plt.scatter( X, Y, color = color )
    
    plt.title( 'X vs Y' )
    
    plt.xlabel( 'X' )
    
    plt.ylabel( 'Y' )
    return fig
def visualize_save(model_name, experiment_name):
    cwd = os.getcwd() 
    script = os.path.realpath(cwd)+"\\"+"savemodel\\" + experiment_name + "\\"+ model_name
    if not os.path.exists(script):
        os.makedirs(script)
    script= script + "\\"+model_name+".png"
    plt.title(model_name)
    plt.savefig(script)
    # plt.show()