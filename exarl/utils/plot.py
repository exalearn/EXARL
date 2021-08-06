from matplotlib import pyplot as plt
import pandas as pd

class Plot(object):

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),\
                (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),\
                (148, 103, 189), (197, 176, 213), (140, 86,75), (196, 156, 148),\
                (227, 119, 194), (247, 182, 210), (127, 127,127), (199, 199, 199),\
                (188, 189, 34), (219, 219, 141), (23, 190,207), (158, 218, 229)]

    tags = ['-*', '-v', '-4', '-o', '-+', '-x', '-s']

    @staticmethod
    def setup_graph(x_width=12, y_width=9, font_size=16):
        plt.figure(figsize=(x_width, y_width))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_bottom()
        ax.yticks(font_size=font_size)
        ax.xticks(font_size=font_size)
        return ax

    @staticmethod
    def plot_multiple_graphs_with_same_x_data(df : pd.Dataframe, x_data:pd.Dataframe,  save_name_path: str, x_title: str, font_size=30, legend_size=24):
        columns = df.columns
        ax = Plot.setup_graph()
        color_index = 0
        for col in columns:
            y_data = df[col]
            t = Plot.tableau20[color_index % len(Plot.tableau20)]
            t2 = tuple(ti/255 for ti in t)
            ax.plot(x_data, y_data, Plot.tags[color_index%len(Plot.tags)], color=t2, markersize=16,label=col)
            color_index += 2
        ax.xlabel(x_title, font_size=font_size)
        ax.ylabel(y_title, font_size=font_size)
        ax.legend(prop={'size': legend_size})
        ax.savefig(save_name_path, bbox_inches='tight')

    @staticmethod
    def plot_multiple_graphs_dict(my_dict,  save_name_path: str, x_title: str, font_size=30, legend_size=24):
        """
        Dictionary: key -> name of approach, data -> {x_data: [], y_data: []}
        """
        columns = df.columns
        ax = Plot.setup_graph()
        color_index = 0
        for key in my_dict:
            try:
                x_data = my_dict[key]["x_data"]
                y_data = my_dict[key]["y_data"]
            except:
                print("The data structure for the dictionary not supported")
                exit()
            t = Plot.tableau20[color_index % len(Plot.tableau20)]
            t2 = tuple(ti/255 for ti in t)
            ax.plot(x_data, y_data, Plot.tags[color_index%len(Plot.tags)], color=t2, markersize=16,label=key)
            color_index += 2
        ax.xlabel(x_title, font_size=font_size)
        ax.ylabel(y_title, font_size=font_size)
        ax.legend(prop={'size': legend_size})
        ax.savefig(save_name_path, bbox_inches='tight')