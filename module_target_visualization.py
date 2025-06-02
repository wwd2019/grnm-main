import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class tf_moudle:
    def __init__(self,tf_list,tf_modules,target_module_ratios,tf_modules_name={}):
        self.tf_list = tf_list
        self.tf_modules = tf_modules
        self.target_module_ratios = target_module_ratios
        self.tf_modules = tf_modules
        self.tf_modules_name = tf_modules_name
    def plot_scatter(self, figsize=(15, 15), marker_size=20, alpha=0.5,
                     xlabel='Transcription Factor Modules', ylabel='Target Module Ratio',
                     title='TF Module vs. Target Module Ratio', save_path=None, color_map=[]):
        """
        Plots a scatter plot of the target module ratios for each transcription factor module,
        with each module displayed in a different color based on the provided color map.

        Parameters:
            figsize (tuple): The dimensions of the figure to be created.
            marker_size (int): The size of the markers in the scatter plot.
            alpha (float): The opacity level of the markers. Ranges from 0 to 1.
            xlabel (str): The label of the x-axis.
            ylabel (str): The label of the y-axis.
            title (str): The title of the plot.
            save_path (str): If provided, the plot will be saved to this path instead of showing interactively.
            color_map (list): A list of colors used to differentiate the modules. Assumes the list length matches the maximum module index.
        """
        # Create a DataFrame for plotting
        tf_module_df = pd.DataFrame({
            'tf_module': self.tf_modules,
            'target_module_ratio': self.target_module_ratios
        })

        # Set default color map if none provided
        if not color_map:
            # Ensure there is a default color for each possible module index
            color_map = sns.color_palette('husl', 1+len(set(self.tf_modules)))

        # Assign colors to each module using direct indexing from color_map
        tf_module_df['colors'] = [color_map[int(i)] for i in self.tf_modules]

        # Create a figure and a set of subplots with custom dimensions
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Scatter plot with direct color mapping from the 'colors' column
        ax.scatter('tf_module', 'target_module_ratio', data=tf_module_df,
                   s=marker_size, color=tf_module_df['colors'], alpha=alpha)

        # Customizing the plot appearance
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # If a save path is provided, save the plot; otherwise, show it
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
