import argparse
from csv import reader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import numpy as np
import csv
from numpy.linalg import norm
from statistics import mean
import json
sns.set()


# xlabel = 'Communication Rounds'
# ylabel ='Test Accuracy'
figsize=(16, 12)


def get_item_from_json_string(string, key="exp_title"):
    try:
        obj = json.loads(string)
        item = obj[key]
    except ValueError:
        item = string
    return item
    

def results_in_df(result_file, window_size=None): 
    """
    The function first converts a metric file to a dataframe with each row 
    in the csv being read in as column in the dataframe. Optionally, it allows 
    for smoothening out the values (column wise)
    """
    df = pd.DataFrame()
    with open(result_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            col_name = get_item_from_json_string(row[0])
            df[col_name] = row[1:]
            df[col_name] = df[col_name].astype(float)
            if window_size: #take column wise rolling averages
                df[col_name] = df[col_name].rolling(window=window_size).mean()
    return df.dropna()


def filter_df(df, cols_to_retain):
    return df[cols_to_retain]


def plot_metrics(metric_file, 
                 window_size=None, 
                 plot_title='', 
                 figsize=figsize, 
                 loc=0, 
                 fontsize=22, 
                 xlim=None, 
                 ylim=None, 
                 stop_at=None, 
                 start_from=None,
                 plot_selected_cols=None,
                 xlabel='Communication Rounds',
                 ylabel='Test Accuracy',
                 save_path=None
                ): 
    """
    This function plots a whole dataframe column by column on a single axis. 
    """
    # create a dataframe from metrics
    df = results_in_df(metric_file, window_size)
    
    # slice accordingly
    df = df[start_from: stop_at]
    
    # if we just want to plot some selected items
    if plot_selected_cols:
        df = filter_df(df, plot_selected_cols)
        
    x_axis = range(len(df))
    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axis
    
    for col in df.columns:
        ax.plot(x_axis, df[col], label=col)
        #print("###########", list(df[col]))
        print(f"{col}: {max(df[col])}, {mean(df[col])}")
        #print(f"{col}: {max(df[col])}, {mean(df[col])}")
        
    # plot annotations
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
    ax.set_title(plot_title, fontsize=fontsize, fontweight='bold')
    ax.legend(loc=loc, fontsize=fontsize, framealpha=0)  # Set legend background transparency
    ax.tick_params(axis='both', labelsize=fontsize)
    if xlim or ylim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    # Set the background color to transparent
    ax.patch.set_alpha(0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_filename', type=str, default='./test_acc.csv', help='path to the targeted metric file')
    parser.add_argument('--save_path', type=str, default='./', help='where to save the plot')
    parser.add_argument('--ylabel', type=str, default='Test Accuracy', help='label for the y-axis')
    parser.add_argument('--xlabel', type=str, default='Communication Rounds', help='label for the x-axis')
    parser.add_argument('--plot_title', type=str, default='Add a custom title', help='title of the plot')
    parser.add_argument('--loc', type=int, default=0, help='where to locate the plot legend')
    parser.add_argument('--window_size', type=int, default=None, help='a window size for smoothening')
    parser.add_argument('--fontsize', type=int, default=22, help='size of the of the txt font')
    parser.add_argument('--stop_at', type=int, default=None, help='where to start the series from')
    parser.add_argument('--start_from', type=int, default=None, help='where to end the series at')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    plot_metrics(args.metric_filename, 
                 window_size=args.window_size, 
                 plot_title=args.plot_title, 
                 figsize=figsize, 
                 loc=args.loc, 
                 fontsize=args.fontsize, 
                 xlim=None, 
                 ylim=None, 
                 stop_at=args.stop_at, 
                 start_from=args.start_from,
                 plot_selected_cols=None,
                 xlabel='Communication Rounds',
                 ylabel='Test Accuracy',
                 save_path=args.save_path
                )
