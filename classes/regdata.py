# import matplotlib.patches as mpatches
import copy
import os
import pickle
import random
import re
import sys
import time
from importlib import reload
from math import sqrt
from typing import Dict, Tuple, Union

import cortex
import h5py
import joblib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import seaborn as sns
import torch
import torchvision.models as models
import yaml
from colorama import Fore, Style
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullFormatter
from nilearn import plotting
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from scipy.special import softmax
from scipy.stats import zscore as zs
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from tqdm import tqdm

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

import funcs.natspatpred
from funcs.analyses import univariate_regression
from classes.utilities import _extract_layno
from funcs.natspatpred import NatSpatPred, VoxelSieve
from unet_recon.inpainting import UNet

NSP = NatSpatPred()
NSP.initialise(verbose=False)


class RegData:
    def __init__(
        self,
        subject: str = "subj01",
        folder: str | None = "unpred",
        model: str = "vgg-b",
        statistic: str = "delta_r",
    ):
        self.subject = subject
        self.folder = folder
        self.model = model
        self.statistic = statistic
        self.cnn_layers = None
        self._build_df(subject, folder, model, statistic)

    # def _build_df(
    #     self,
    #     subject: str,
    #     folder: str,
    #     model: str,
    #     statistic: str,
    #     main_df: bool = True,
    # ):
    #     # Directory containing the CSV files
    #     directory = f"{NSP.own_datapath}/{subject}/results/{folder}/"

    #     # List of filenames in the directory
    #     filenames = os.listdir(directory)

    #     # Empty DataFrame to store the results
    #     df = pd.DataFrame()

    #     # Variable to store the 'roi' column from the first file
    #     roi_column = None

    #     for filename in filenames:
    #         # Check if the filename starts with the model name
    #         if filename.startswith(model) and filename.endswith(".csv"):
    #             # Get the layer number from the filename
    #             layno = NSP.utils.get_layer_file(filename, "lay")

    #             # Read the CSV file into a DataFrame
    #             file_df = pd.read_csv(os.path.join(directory, filename))

    #             # Store the 'roi' column from the first file
    #             if roi_column is None and "roi" in file_df.columns:
    #                 roi_column = file_df["roi"]

    #             # Select the statistic column and rename it
    #             file_df = file_df[[statistic]].rename(
    #                 columns={statistic: f"{statistic}_{layno+1}"}
    #             )

    #             # Add the column to the result DataFrame
    #             if df.empty:
    #                 df = file_df
    #             else:
    #                 df = pd.concat([df, file_df], axis=1)

    #     # Sort the remaining columns
    #     df = df.sort_index(
    #         axis=1,
    #         key=lambda x: x.map(lambda y: int(re.findall(f"{statistic}_(\d+)", y)[0])),
    #     )

    #     # Add the 'roi' column to the DataFrame
    #     if roi_column is not None:
    #         df.insert(0, "roi", roi_column)

    #     if main_df:
    #         self.df = df
    #         self.cnn_layers = list(range(1, len(self.df.columns[1:]) + 1))
    #     else:
    #         return df
        
    def _build_df(
        self,
        subject: str,
        folder: str,
        model: str,
        statistic: str,
        main_df: bool = True,
        add_xyz: bool = True
    ):
        # Directory containing the CSV files
        directory = f"{NSP.own_datapath}/{subject}/results/{folder}/"

        # List of filenames in the directory
        filenames = os.listdir(directory)

        # Empty DataFrame to store the results
        df = pd.DataFrame()

        # Variable to store the 'roi' column from the first file
        roi_column = None
        xyz_columns = None
        
        for filename in filenames:
            # Check if the filename starts with the model name
            if filename.startswith(model) and filename.endswith(".csv"):
                # Get the layer number from the filename
                layno = NSP.utils.get_layer_file(filename, "lay")

                # Read the CSV file into a DataFrame
                file_df = pd.read_csv(os.path.join(directory, filename))

                # Store the 'roi' column from the first file
                if roi_column is None and "roi" in file_df.columns:
                    roi_column = file_df["roi"]

                if xyz_columns is None and "x" in file_df.columns and "y" in file_df.columns and "z" in file_df.columns:
                    xyz_columns = file_df[["x", "y", "z"]]
                
                # Select the statistic column and rename it
                file_df = file_df[[statistic]].rename(
                    columns={statistic: f"{statistic}_{layno+1}"}
                )

                # Add the column to the result DataFrame
                if df.empty:
                    df = file_df
                else:
                    df = pd.concat([df, file_df], axis=1)

        # Sort the remaining columns
        df = df.sort_index(
            axis=1,
            key=lambda x: x.map(lambda y: int(re.findall(f"{statistic}_(\d+)", y)[0])),
        )

        # Add the 'roi' column to the DataFrame
        if roi_column is not None:
            df.insert(0, "roi", roi_column)
        # Add the xyz columns to the DataFrame
        if xyz_columns is not None and add_xyz:
            df = pd.concat([xyz_columns, df], axis=1)
        
        # if main_df:
        #     self.df = df
        #     self.cnn_layers = list(range(1, len(self.df.columns[1:]) + 1))
        # else:
        if main_df:
            self.df = df
            # Find the indices of the columns that start with the statistic name
            self.cnn_layers = [i for i, col in enumerate(self.df.columns) if col.startswith(statistic)]
        else:
    
            return df

    
        
    def _stat_to_nifti(self, max_or_weighted: str = "weighted", verbose: bool = True):
        pass
    
    def _zscore(self, verbose: bool = True, copy_df: bool | pd.DataFrame = False):
        """
        Z-score the values in the DataFrame.
        """
        data = self.df if copy_df is False else copy_df

        data.iloc[:, self.cnn_layers] = zs(data.iloc[:, self.cnn_layers])
        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Values z-scored. Check whether this is necessary for the current statistic."
            )

        return data if copy_df is not False else None

    def _normalize_per_voxel(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):
        """
        Normalize the values by the maximum value for each voxel (row), ensuring that the minimum value is 0.
        """
        data = self.df if copy_df is False else copy_df

        df_min = data.iloc[:, self.cnn_layers].min(axis=1)
        # df_max = data.iloc[:, 1:].max(axis=1)
        df_max = data.iloc[:, self.cnn_layers].max(axis=1)
        
        data.iloc[:, self.cnn_layers] = (
            data.iloc[:, self.cnn_layers]
            .sub(df_min, axis=0)
            .div(df_max - df_min, axis=0)
        )
        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Values normalised by dividing by the maximum value for each voxel (row), min values capped at 0."
            )

        return data if copy_df is not False else None

    def _scale_to_baseline(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):
        """
        Scale the values to the baseline value for each ROI.
        NOTE: This method is not recommended for the delta-R statistic. As the delta-R value is already
        a product of two correlation values that are standardized already, it is not necessary to scale
        although it might seem intuitively necessary. Subtracting the baseline value from the delta-R value
        is a balance between simplicity and interpretability.
        """
        data = self.df if copy_df is False else copy_df

        baseline_data = self._build_df(
            self.subject,
            self.folder,
            self.model,
            statistic="R_alt_model",
            main_df=False,
        ).iloc[:, self.cnn_layers[0]]

        scaled_data = (
            data.iloc[:, self.cnn_layers]
            .div(baseline_data, axis=0)
            .mul(1 - baseline_data.pow(2), axis=0)
        )

        data.iloc[:, self.cnn_layers] = scaled_data

        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Values scaled to the baseline value for each ROI."
            )

        return data if copy_df is not False else None

    def _weigh_mean_layer(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):

        data = self.df if copy_df is False else copy_df

        # Normalize each row by the sum of its values
        df_normalized = data.iloc[:, self.cnn_layers].div(
            data.iloc[:, self.cnn_layers].sum(axis=1), axis=0
        )

        # Multiply each value by its column index
        weights = [(_extract_layno(col)) for col in df_normalized.columns]
        # if verbose:
        #     print(f"These are the weights for each column: {weights}")

        df_weighted = df_normalized.mul(weights, axis=1)

        # Calculate the mean of the weighted values and add it as a new column
        data["Mean Weighted Layer"] = round(df_weighted.mean(axis=1) * max(weights), 2)
        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Added weighted means of the relative CNN-layer delta-R ranking to the DataFrame."
            )

        return data if copy_df is not False else None

    def _get_max_layer(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):
        """
        Get the maximum layer index for each row in the DataFrame. Corresponds with the CNN-layer that has
        the highest delta-R value for each ROI, or any other statistic.

        Parameters:
        - verbose (bool): If True, print a message indicating that the maximum layer index has been added to the DataFrame.

        Returns:
        None
        """
        data = self.df if copy_df is False else copy_df

        # Get the index of the maximum value in each row
        max_indices = (
            np.argmax(data.values[:, self.cnn_layers], axis=1) + 1
        )  # Add 1 to the max_indices to get the layer number

        assign_str = "Max Layer"
        # Add the max_indices as a new column
        data[assign_str] = max_indices

        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Added the maximum layer index to the DataFrame."
            )

        return data if copy_df is not False else None

    def assign_layers(self, max_or_weighted: str = "weighted", verbose: bool = True, title: str = None):
        """
        Assigns layers to each ROI based on the maximum value in each row of a DataFrame.

        Parameters:
        - results (pd.DataFrame): The DataFrame containing the results.
        - max_or_weighted (str): The method to use for assigning layers. Default is 'max', other option is 'weighted'.

        Returns:
            dataframe: The DataFrame with the '{assign_str} Layer' column added.
        """

        df = self.df.copy()
        print(
            "Using a copy of the DataFrame for layer assignment, the original DataFrame will not be changed."
        )
        df = self._normalize_per_voxel(verbose=verbose, copy_df=df)

        if max_or_weighted == "max":
            df = self._get_max_layer(verbose=verbose, copy_df=df)
            assign_str = "Max Layer"

        elif max_or_weighted == "weighted":
            df = self._weigh_mean_layer(verbose=verbose, copy_df=df)
            assign_str = "Mean Weighted Layer"

        lay_colours = max(self.cnn_layers) - self.cnn_layers[0] + 1


        barcmap = LinearSegmentedColormap.from_list(
            "NavyBlueVeryLightGreyDarkRed",
            ["#000080", "#CCCCCC", "#FFA500", "#FF0000"],
            N=lay_colours,
        )

        # Calculate the proportions of max_indices within each ROI
        df_prop = (
            df.groupby("roi")[assign_str]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # Plot the proportions using a stacked bar plot
        ax = df_prop.plot(kind="bar", stacked=True, colormap=barcmap, edgecolor="none")

        # Add a y-axis label
        ax.set_ylabel("Layer assignment (%)")

        leg_colours = [
            patches.Patch(
                # color=barcmap(i / (len(self.cnn_layers) - 1)), label=str(layer)
                color=barcmap(i / (len(self.cnn_layers) - 1)), label=str(layer - self.cnn_layers[0] + 1)
                # color=barcmap(i / (len(self.cnn_layers) - self.cnn_layers[0])), label=str(layer)
            )
            for i, layer in enumerate(self.cnn_layers)
        ]

        # Create legend
        legend = plt.legend(
            handles=leg_colours,
            title="CNN\nLayer",
            loc="center right",
            bbox_to_anchor=(1.15, 0.5),
            ncol=1,
            fancybox=False,
            shadow=False,
            fontsize=10,
        )
        if title is None:
            plt.title(f"Layer assignment {self.folder} {self.model} {self.statistic} {max_or_weighted}")
        else:
            plt.title(title)
            
        plt.show()

    def mean_lines(
            self, fit_polynom: bool = False, polynom_order: int = 2, verbose: bool = True, plot_catplot: bool = True
        ):
            """
            Plots the mean values of each ROI across layers.

            Parameters:
            - results (pd.DataFrame): The DataFrame containing the results.

            Returns:
                None
            """
            df = self.df.copy()

            # Assuming df is your DataFrame
            present_id_vars = ("x", "y", "z", "roi", "Max Layer", "Mean Weighted Layer")

            # Create a tuple of column names that exist in both the DataFrame and the tuple
            present_id_vars = tuple(col for col in present_id_vars if col in df.columns)

            # Reshape the DataFrame
            df_melted = df.melt(
                id_vars=present_id_vars, var_name="column", value_name="delta_r"
            )

            # Extract numeric part from 'column' and convert to numeric
            df_melted["column"] = df_melted["column"].str.extract("(\d+)").astype(int) - 1
            df_melted = df_melted.sort_values(by="column")

            rois = sorted(df_melted["roi"].unique(), key=lambda x: int(x.split("V")[1]))

            # Create a color palette
            palette = sns.color_palette(n_colors=len(rois))

            # Create a dictionary mapping ROIs to colors
            roi_to_color = dict(zip(rois, palette))

            if plot_catplot:
                # Create a catplot with alpha set to 0.01
                catplot = sns.catplot(
                    data=df_melted,
                    x="column",
                    y="delta_r",
                    hue="roi",
                    jitter=True,
                    palette=roi_to_color,
                    alpha=0.01,
                    legend=False,
                )
            else:
                # Create a figure and axes to plot on if not plotting catplot
                fig, ax = plt.subplots()

            # Create a regplot and a lineplot for each ROI
            for i, roi in enumerate(rois):
                roi_data = df_melted[df_melted["roi"] == roi]
                if fit_polynom:
                    sns.regplot(
                        data=roi_data,
                        x="column",
                        y="delta_r",
                        scatter=False,
                        truncate=False,
                        order=polynom_order,
                        color=roi_to_color[roi],
                        ax=catplot.ax if plot_catplot else ax,
                    )
                else:
                    sns.lineplot(
                        data=roi_data,
                        x="column",
                        y="delta_r",
                        color=roi_to_color[roi],
                        ax=catplot.ax if plot_catplot else ax,
                    )

            # Set the lower limit of the y-axis to 0
            (catplot.ax if plot_catplot else ax).set_ylim(bottom=0)
            (catplot.ax if plot_catplot else ax).set_ylabel("ΔR Value")
            (catplot.ax if plot_catplot else ax).set_xlabel("CNN Layer")
            (catplot.ax if plot_catplot else ax).set_xticks(range(len(self.cnn_layers)))
            (catplot.ax if plot_catplot else ax).set_xticklabels([f"{i+1}" for i in range(len(self.cnn_layers))])

            # Manually add the legend
            (catplot.ax if plot_catplot else ax).legend(
                handles=[patches.Patch(color=roi_to_color[roi], label=roi) for roi in rois]
            )
            plt.show()