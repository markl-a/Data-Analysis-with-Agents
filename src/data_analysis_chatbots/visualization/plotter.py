"""Plotting and visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from loguru import logger


class Plotter:
    """Create visualizations for data analysis."""

    def __init__(
        self,
        style: str = 'seaborn-v0_8',
        palette: str = 'husl',
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 100
    ):
        """
        Initialize the Plotter.

        Args:
            style: Matplotlib style
            palette: Seaborn color palette
            figure_size: Default figure size
            dpi: DPI for plots
        """
        self.style = style
        self.palette = palette
        self.figure_size = figure_size
        self.dpi = dpi

        # Set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not found. Using default.")
            plt.style.use('default')

        sns.set_palette(palette)

    def plot_distribution(
        self,
        data: pd.Series,
        title: str = "Distribution Plot",
        xlabel: str = None,
        bins: int = 30,
        kde: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of a variable.

        Args:
            data: Data to plot
            title: Plot title
            xlabel: X-axis label
            bins: Number of bins for histogram
            kde: Whether to show KDE
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        sns.histplot(data, bins=bins, kde=kde, ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_scatter(
        self,
        x: pd.Series,
        y: pd.Series,
        hue: Optional[pd.Series] = None,
        title: str = "Scatter Plot",
        xlabel: str = None,
        ylabel: str = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a scatter plot.

        Args:
            x: X-axis data
            y: Y-axis data
            hue: Variable for color encoding
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        if hue is not None:
            sns.scatterplot(x=x, y=y, hue=hue, palette=self.palette, s=100, alpha=0.7, ax=ax)
        else:
            sns.scatterplot(x=x, y=y, s=100, alpha=0.7, ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)

        if hue is not None:
            ax.legend(title=hue.name, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_clusters(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        cluster_col: str,
        centers: Optional[pd.DataFrame] = None,
        title: str = "Customer Segments",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot clustering results.

        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            cluster_col: Column with cluster labels
            centers: DataFrame with cluster centers
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Plot clusters
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=cluster_col,
            palette=self.palette,
            s=100,
            alpha=0.7,
            ax=ax
        )

        # Plot centers if provided
        if centers is not None and x_col in centers.columns and y_col in centers.columns:
            ax.scatter(
                centers[x_col],
                centers[y_col],
                s=300,
                c='red',
                marker='X',
                edgecolors='black',
                linewidths=2,
                label='Centers',
                zorder=10
            )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_elbow(
        self,
        k_values: List[int],
        inertias: List[float],
        title: str = "Elbow Method for Optimal K",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot elbow curve for K-means.

        Args:
            k_values: List of K values
            inertias: List of inertia values
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Inertia', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_rfm_heatmap(
        self,
        rfm_df: pd.DataFrame,
        title: str = "RFM Correlation Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot RFM correlation heatmap.

        Args:
            rfm_df: DataFrame with RFM metrics
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)

        # Select only RFM columns
        rfm_cols = [col for col in ['Recency', 'Frequency', 'Monetary'] if col in rfm_df.columns]

        if not rfm_cols:
            logger.error("No RFM columns found in DataFrame")
            return fig

        # Calculate correlation
        corr = rfm_df[rfm_cols].corr()

        # Plot heatmap
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(title, fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_segment_distribution(
        self,
        segments: pd.Series,
        title: str = "Customer Segment Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of customer segments.

        Args:
            segments: Series with segment labels
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)

        # Count plot
        segment_counts = segments.value_counts()
        colors = sns.color_palette(self.palette, len(segment_counts))

        ax1.bar(range(len(segment_counts)), segment_counts.values, color=colors)
        ax1.set_xticks(range(len(segment_counts)))
        ax1.set_xticklabels(segment_counts.index, rotation=45, ha='right')
        ax1.set_title('Segment Counts', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Customers', fontsize=12)

        # Pie chart
        ax2.pie(
            segment_counts.values,
            labels=segment_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax2.set_title('Segment Percentage', fontsize=14, fontweight='bold')

        fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_comparison(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        group_col: str,
        plot_type: str = 'box',
        title: str = "Comparison Plot",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison across groups.

        Args:
            df: DataFrame with data
            x_col: Column for x-axis (groups)
            y_col: Column for y-axis (values)
            group_col: Column for grouping
            plot_type: Type of plot ('box', 'violin', 'bar')
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        if plot_type == 'box':
            sns.boxplot(data=df, x=x_col, y=y_col, hue=group_col, palette=self.palette, ax=ax)
        elif plot_type == 'violin':
            sns.violinplot(data=df, x=x_col, y=y_col, hue=group_col, palette=self.palette, ax=ax)
        elif plot_type == 'bar':
            sns.barplot(data=df, x=x_col, y=y_col, hue=group_col, palette=self.palette, ax=ax)
        else:
            logger.error(f"Unknown plot type: {plot_type}")
            return fig

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)

        if group_col:
            ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig
