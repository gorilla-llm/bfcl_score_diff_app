import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import tempfile
import zipfile
import shutil
from io import BytesIO


class VersionComparisonAnalyzer:
    def __init__(self, old_file, new_file):
        self.old_df = pd.read_csv(old_file)
        self.new_df = pd.read_csv(new_file)
        self.merged_df = None
        self.summary_categories = ["Non-Live AST Acc", "Live Acc", "Multi Turn Acc"]
        self.subcategories = [
            "Non-Live Simple AST",
            "Non-Live Multiple AST",
            "Non-Live Parallel AST",
            "Non-Live Parallel Multiple AST",
            "Non-Live Exec Acc",
            "Non-Live Simple Exec",
            "Non-Live Multiple Exec",
            "Non-Live Parallel Exec",
            "Non-Live Parallel Multiple Exec",
            "Live Simple AST",
            "Live Multiple AST",
            "Live Parallel AST",
            "Live Parallel Multiple AST",
            "Multi Turn Base",
            "Multi Turn Miss Func",
            "Multi Turn Miss Param",
            "Multi Turn Long Context",
        ]
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join(
            "scores_analytics", f"comparison_results_{timestamp}"
        )
        self.version_dir = os.path.join(self.base_dir, "version_summary")
        self.detailed_dir = os.path.join(self.base_dir, "detailed_analysis")
        os.makedirs(self.version_dir, exist_ok=True)
        os.makedirs(self.detailed_dir, exist_ok=True)

    def get_version_path(self, filename):
        return os.path.join(self.version_dir, filename)

    def get_detailed_path(self, filename):
        return os.path.join(self.detailed_dir, filename)

    def merge_datasets(self):
        """Merge old and new datasets based on Model column and track added/removed models"""
        # Store models unique to each dataset
        self.old_only_models = set(self.old_df["Model"]) - set(self.new_df["Model"])
        self.new_only_models = set(self.new_df["Model"]) - set(self.old_df["Model"])

        # Merge only models that exist in both datasets
        self.merged_df = pd.merge(
            self.old_df,
            self.new_df,
            on="Model",
            suffixes=("_old", "_new"),
            how="inner",  # Only keep models that exist in both datasets
        )
        return self.merged_df

    def calculate_differences(self):
        """Calculate differences for all metrics"""
        diff_cols = {}
        for col in self.summary_categories + self.subcategories:
            old_col = col + "_old"
            new_col = col + "_new"
            if old_col in self.merged_df.columns and new_col in self.merged_df.columns:
                diff_col = col + "_diff"
                self.merged_df[diff_col] = self.merged_df[new_col].str.rstrip(
                    "%"
                ).astype(float) - self.merged_df[old_col].str.rstrip("%").astype(float)
                diff_cols[col] = diff_col
        return diff_cols

    def generate_summary_report(self):
        """Generate summary report focusing on major changes"""
        report = []
        report.append("# Version Comparison Report\n")

        # Report added and removed models
        if self.new_only_models:
            report.append("## New Models Added")
            for model in sorted(self.new_only_models):
                report.append(f"- {model}")
            report.append("")  # Empty line for spacing

        if self.old_only_models:
            report.append("## Models Removed")
            for model in sorted(self.old_only_models):
                report.append(f"- {model}")
            report.append("")  # Empty line for spacing

        # Rank changes
        rank_changes = self.merged_df.apply(
            lambda x: -1 * (x["Rank_new"] - x["Rank_old"]), axis=1
        )
        significant_rank_changes = rank_changes[abs(rank_changes) >= 5]

        report.append("## Significant Rank Changes (≥5 positions)\n")
        report.append("| Model | Change | Direction |")
        report.append("|-------|--------|-----------|")
        for model in significant_rank_changes.index:
            change = significant_rank_changes[model]
            direction = "⬆️ Improved" if change < 0 else "⬇️ Dropped"
            report.append(
                f"| {self.merged_df.loc[model, 'Model']} | {abs(change)} | {direction} |"
            )

        # Export rank changes to CSV
        rank_change_df = pd.DataFrame(
            {
                "Model": self.merged_df["Model"],
                "Old Rank": self.merged_df["Rank_old"],
                "New Rank": self.merged_df["Rank_new"],
                "Rank Change": rank_changes,
            }
        )
        rank_change_df.to_csv(self.get_version_path("rank_changes.csv"), index=False)

        return "\n".join(report)

    def plot_summary_changes(self):
        """Create visualizations for major metric changes"""
        plt.figure(figsize=(15, 10))

        for i, category in enumerate(self.summary_categories):
            plt.subplot(3, 1, i + 1)
            diff_col = category + "_diff"
            if diff_col in self.merged_df.columns:
                abs_diff = pd.to_numeric(
                    self.merged_df[diff_col], errors="coerce"
                ).abs()
                data = self.merged_df.loc[abs_diff.nlargest(10).index][
                    [diff_col, "Model"]
                ]
                data[diff_col] = pd.to_numeric(data[diff_col], errors="coerce")

                bars = plt.barh(
                    data["Model"],
                    data[diff_col],
                    color=["#FF6B6B" if x < 0 else "#4CAF50" for x in data[diff_col]],
                )

                for bar in bars:
                    width = bar.get_width()
                    plt.text(
                        width,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.1f}%",
                        ha="left" if width >= 0 else "right",
                        va="center",
                    )

                plt.title(f"Top Changes in {category}", pad=20)
                plt.xlabel("Percentage Point Change")
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.get_version_path("summary_changes.png"), bbox_inches="tight", dpi=300
        )
        plt.close()

    def generate_detailed_report(self):
        """Generate detailed analysis report"""
        report = []
        report.append("# Detailed Version Comparison Analysis\n")

        # Add timestamp and file information
        report.append(
            f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        report.append("Comparing versions:")
        report.append(f"- Old version: data_overall_10_21.csv")
        report.append(f"- New version: data_overall_11_9.csv\n")

        # Add summary statistics for each category
        report.append("## Summary Category Changes\n")
        for category in self.summary_categories:
            diff_col = f"{category}_diff"
            if diff_col in self.merged_df.columns:
                diffs = pd.to_numeric(self.merged_df[diff_col], errors="coerce")
                report.append(f"### {category}")
                report.append(f"- Average change: {diffs.mean():.2f}%")
                report.append(f"- Models improved: {(diffs > 0).sum()}")
                report.append(f"- Models declined: {(diffs < 0).sum()}")
                report.append(f"- Largest improvement: {diffs.max():.2f}%")
                report.append(f"- Largest decline: {diffs.min():.2f}%\n")

        # Add subcategory analysis
        report.append("## Subcategory Analysis\n")
        for subcategory in self.subcategories:
            diff_col = f"{subcategory}_diff"
            if diff_col in self.merged_df.columns:
                diffs = pd.to_numeric(self.merged_df[diff_col], errors="coerce")
                report.append(f"### {subcategory}")
                report.append(f"- Average change: {diffs.mean():.2f}%")
                report.append(f"- Standard deviation: {diffs.std():.2f}%")
                report.append(
                    f"- Models showing significant change (>5%): {(abs(diffs) > 5).sum()}\n"
                )

        # Add notable model changes
        report.append("## Notable Model Changes\n")
        if "Overall Acc_diff" in self.merged_df.columns:
            overall_diff = pd.to_numeric(
                self.merged_df["Overall Acc_diff"], errors="coerce"
            )
            top_improved = self.merged_df[overall_diff > 0].nlargest(
                5, "Overall Acc_diff"
            )
            top_declined = self.merged_df[overall_diff < 0].nsmallest(
                5, "Overall Acc_diff"
            )

            report.append("### Top 5 Most Improved Models")
            for _, row in top_improved.iterrows():
                report.append(f"- {row['Model']}: +{row['Overall Acc_diff']:.2f}%")

            report.append("\n### Top 5 Most Declined Models")
            for _, row in top_declined.iterrows():
                report.append(f"- {row['Model']}: {row['Overall Acc_diff']:.2f}%")

        # Return the complete report as a string
        return "\n".join(report)

    def generate_heatmap(self):
        """Generate heatmap of changes across all metrics"""
        diff_columns = [col for col in self.merged_df.columns if col.endswith("_diff")]
        diff_data = self.merged_df[["Model"] + diff_columns].set_index("Model")

        for col in diff_columns:
            diff_data[col] = pd.to_numeric(diff_data[col], errors="coerce")

        # Calculate figure size based on number of models and metrics
        n_models = len(diff_data)
        n_metrics = len(diff_columns)

        # Adjust figure size to accommodate all models
        plt.figure(figsize=(n_metrics * 0.4 + 4, n_models * 0.25 + 2))

        # Create heatmap with adjusted parameters
        heatmap = sns.heatmap(
            diff_data,
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 7},  # Smaller annotation font size
            cbar_kws={"label": "Percentage Point Change"},
            yticklabels=True,
        )  # Ensure y-labels (model names) are shown

        # Adjust y-axis labels (model names)
        plt.yticks(rotation=0, fontsize=8)  # Horizontal model names with smaller font
        plt.xticks(
            rotation=45, ha="right", fontsize=8
        )  # Angled metric names with smaller font

        plt.title(
            "Changes Heatmap Across All Metrics\n(Green: Improvement, Red: Decline)",
            pad=20,
            fontsize=10,
        )

        # Add explanation text with smaller font
        plt.figtext(
            0.1,
            -0.1,
            "How to interpret: Each cell shows the percentage point change between versions.\n"
            + "Positive values (green) indicate improvement, negative values (red) indicate decline.\n"
            + "Darker colors represent larger changes.",
            wrap=True,
            horizontalalignment="left",
            fontsize=8,
        )

        plt.tight_layout()
        plt.savefig(
            self.get_detailed_path("changes_heatmap.png"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.5,
        )
        plt.close()

    def export_detailed_statistics(self):
        """Export detailed statistics to CSV"""
        stats_data = []

        for col in self.summary_categories + self.subcategories:
            diff_col = f"{col}_diff"
            if diff_col in self.merged_df.columns:
                diff_values = pd.to_numeric(self.merged_df[diff_col], errors="coerce")
                stats = {
                    "Metric": col,
                    "Mean Change": diff_values.mean(),
                    "Median Change": diff_values.median(),
                    "Std Dev": diff_values.std(),
                    "Max Improvement": diff_values.max(),
                    "Max Decline": diff_values.min(),
                    "Models Improved": (diff_values > 0).sum(),
                    "Models Declined": (diff_values < 0).sum(),
                }
                stats_data.append(stats)

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(self.get_detailed_path("metric_statistics.csv"), index=False)

    def generate_summary_csv(self):
        """Generate a summary CSV with all metrics"""
        summary_dir = os.path.join(self.base_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        # Prepare columns for all metrics
        metrics = self.summary_categories + self.subcategories
        columns = []
        for metric in metrics:
            columns.extend([f"{metric}_old", f"{metric}_new", f"{metric}_diff"])

        # Create summary dataframe
        summary_df = self.merged_df[["Model"] + columns].copy()

        # Convert percentage strings to numbers
        for col in columns:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(
                    (
                        summary_df[col].str.rstrip("%")
                        if col.endswith(("_old", "_new"))
                        else summary_df[col]
                    ),
                    errors="coerce",
                )

        # Save to CSV
        summary_df.to_csv(
            os.path.join(summary_dir, "all_metrics_comparison.csv"), index=False
        )

    def generate_category_analysis(self):
        """Generate comprehensive analysis (charts, heatmaps, CSVs) for each category"""
        # Create directory structure
        base_categories_dir = os.path.join(self.detailed_dir, "categories")
        for category in ["charts", "heatmaps", "data"]:
            os.makedirs(os.path.join(base_categories_dir, category), exist_ok=True)

        # Process each main category
        for main_category in self.summary_categories:
            # Get related metrics for this category
            related_metrics = [
                m for m in self.subcategories if m.startswith(main_category.split()[0])
            ]
            metrics = [main_category] + related_metrics

            # 1. Generate CSV with all data
            csv_data = self.merged_df[
                ["Model"]
                + [f"{m}_old" for m in metrics]
                + [f"{m}_new" for m in metrics]
                + [f"{m}_diff" for m in metrics]
            ].copy()

            # Convert percentage strings to numeric
            for col in csv_data.columns:
                if col != "Model":
                    if col.endswith(("_old", "_new")):
                        csv_data[col] = pd.to_numeric(
                            csv_data[col].str.rstrip("%"), errors="coerce"
                        )
                    else:
                        csv_data[col] = pd.to_numeric(csv_data[col], errors="coerce")

            # Save CSV
            csv_path = os.path.join(
                base_categories_dir,
                "data",
                f'{main_category.lower().replace(" ", "_")}_full_analysis.csv',
            )
            csv_data.to_csv(csv_path, index=False)

            # 2. Generate Heatmap
            plt.figure(figsize=(len(metrics) * 1.5, len(self.merged_df) * 0.3))

            # Prepare heatmap data (differences only)
            heatmap_data = csv_data[[f"{m}_diff" for m in metrics]].copy()
            heatmap_data.index = csv_data["Model"]

            # Create heatmap
            sns.heatmap(
                heatmap_data,
                cmap="RdYlGn",
                center=0,
                annot=True,
                fmt=".1f",
                cbar_kws={"label": "Percentage Point Change"},
                xticklabels=[
                    m.replace(main_category.split()[0] + " ", "") for m in metrics
                ],
            )

            plt.title(
                f"{main_category} Changes Heatmap\n(Green: Improvement, Red: Decline)"
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save heatmap
            heatmap_path = os.path.join(
                base_categories_dir,
                "heatmaps",
                f'{main_category.lower().replace(" ", "_")}_heatmap.png',
            )
            plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
            plt.close()

            # 3. Generate Detailed Charts
            for metric in metrics:
                old_col = f"{metric}_old"
                new_col = f"{metric}_new"
                diff_col = f"{metric}_diff"

                plt.figure(figsize=(15, 10))

                # Sort by absolute difference
                plot_data = csv_data.copy()
                plot_data["abs_diff"] = abs(plot_data[diff_col])
                plot_data = plot_data.sort_values("abs_diff", ascending=False)

                x = np.arange(len(plot_data))
                width = 0.35

                # Create bars
                plt.bar(
                    x - width / 2,
                    plot_data[old_col],
                    width,
                    label="Old Score",
                    color="lightgray",
                )
                plt.bar(
                    x + width / 2,
                    plot_data[new_col],
                    width,
                    label="New Score",
                    color="lightblue",
                )

                # Add difference annotations
                for i, (_, row) in enumerate(plot_data.iterrows()):
                    diff = row[diff_col]
                    if pd.notna(diff):
                        color = "green" if diff > 0 else "red"
                        plt.annotate(
                            f"{diff:+.1f}%",
                            xy=(i, max(row[old_col], row[new_col]) + 1),
                            ha="center",
                            va="bottom",
                            color=color,
                        )

                plt.xlabel("Models")
                plt.ylabel("Score (%)")
                plt.title(
                    f"Score Comparison for {metric}\n(All Models, Sorted by Change Magnitude)"
                )
                plt.xticks(x, plot_data["Model"], rotation=90)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save chart
                chart_path = os.path.join(
                    base_categories_dir,
                    "charts",
                    f'{metric.lower().replace(" ", "_")}_full_comparison.png',
                )
                plt.savefig(chart_path, bbox_inches="tight", dpi=300)
                plt.close()

    def generate_table_heatmaps(self):
        """Generate table-style heatmaps showing old, new scores and differences"""
        # Create directory for table heatmaps
        table_heatmaps_dir = os.path.join(self.detailed_dir, "table_heatmaps")
        os.makedirs(table_heatmaps_dir, exist_ok=True)

        # Process each category
        categories = self.summary_categories + self.subcategories
        for category in categories:
            old_col = f"{category}_old"
            new_col = f"{category}_new"
            diff_col = f"{category}_diff"

            if all(
                col in self.merged_df.columns for col in [old_col, new_col, diff_col]
            ):
                # Prepare data
                plot_data = self.merged_df[["Model", old_col, new_col, diff_col]].copy()

                # Convert values to numeric
                for col in [old_col, new_col]:
                    if plot_data[col].dtype == object:
                        plot_data[col] = pd.to_numeric(
                            plot_data[col].str.rstrip("%"), errors="coerce"
                        )
                plot_data[diff_col] = pd.to_numeric(
                    plot_data[diff_col], errors="coerce"
                )

                # Calculate percent change with handling for edge cases
                def safe_pct_change(old, new):
                    if pd.isna(old) or pd.isna(new) or old == 0:
                        return np.nan
                    return (new - old) / old * 100

                plot_data["pct_change"] = plot_data.apply(
                    lambda row: safe_pct_change(row[old_col], row[new_col]), axis=1
                )

                # Sort by absolute difference
                plot_data["abs_diff"] = abs(plot_data[diff_col])
                plot_data = plot_data.sort_values("abs_diff", ascending=False)
                plot_data = plot_data.drop("abs_diff", axis=1)

                # Create figure with custom dimensions
                fig, ax = plt.subplots(figsize=(15, len(plot_data) * 0.3 + 2))
                ax.axis("tight")
                ax.axis("off")

                # Prepare cell colors based on diff values
                diff_values = plot_data[diff_col].values
                max_abs_diff = max(
                    abs(np.nanmin(diff_values)), abs(np.nanmax(diff_values))
                )

                # Create color array (5 columns to match table data)
                colors = np.zeros((len(plot_data), 5, 3))  # RGBA for each cell

                # Set colors for diff columns
                norm = plt.Normalize(-max_abs_diff, max_abs_diff)
                cmap = plt.cm.RdYlGn  # Red for negative, Green for positive
                diff_colors = cmap(norm(diff_values))

                # Fill color array
                colors[:, 0] = [0.95, 0.95, 0.95]  # Light gray for model names
                colors[:, 1] = [0.95, 0.95, 0.95]  # Light gray for old scores
                colors[:, 2] = [0.95, 0.95, 0.95]  # Light gray for new scores
                colors[:, 3] = diff_colors[:, :3]  # Color based on diff
                colors[:, 4] = diff_colors[
                    :, :3
                ]  # Color based on diff for percent change

                # Format cell values with proper handling of NaN
                def format_pct(x, include_plus=False):
                    if pd.isna(x):
                        return "N/A"
                    if include_plus:
                        return f"{'+' if x > 0 else ''}{x:.1f}%"
                    return f"{x:.1f}%"

                # Create table data
                table_data = np.column_stack(
                    [
                        plot_data["Model"],
                        [format_pct(x) for x in plot_data[old_col]],
                        [format_pct(x) for x in plot_data[new_col]],
                        [format_pct(x, True) for x in plot_data[diff_col]],
                        [format_pct(x, True) for x in plot_data["pct_change"]],
                    ]
                )

                table = ax.table(
                    cellText=table_data,
                    colLabels=[
                        "Model",
                        "Old Score",
                        "New Score",
                        "Abs Change",
                        "Rel Change (%)",
                    ],
                    cellColours=colors,
                    cellLoc="center",
                    loc="center",
                    colWidths=[0.3, 0.15, 0.15, 0.15, 0.15],
                )

                # Adjust table style
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)

                # Add title
                plt.title(
                    f"{category} Score Changes\n(Sorted by Change Magnitude)", pad=20
                )

                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.2)
                cbar.set_label("Percentage Point Change")

                plt.tight_layout()

                # Save figure
                fig_path = os.path.join(
                    table_heatmaps_dir,
                    f'{category.lower().replace(" ", "_")}_table_heatmap.png',
                )
                plt.savefig(fig_path, bbox_inches="tight", dpi=300, pad_inches=0.5)
                plt.close()


def main():
    st.set_page_config(page_title="CSV Version Comparison Tool", layout="wide")

    st.title("CSV Version Comparison Tool")
    st.write("Upload two CSV files to compare metrics and generate analysis reports")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Old Version CSV")
        old_file = st.file_uploader("Upload old version CSV", type=["csv"])

    with col2:
        st.subheader("New Version CSV")
        new_file = st.file_uploader("Upload new version CSV", type=["csv"])

    if old_file and new_file:
        if st.button("Process Files", type="primary"):
            with st.spinner("Processing files and generating analysis..."):
                try:
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded files
                        old_path = os.path.join(temp_dir, "old_file.csv")
                        new_path = os.path.join(temp_dir, "new_file.csv")

                        with open(old_path, "wb") as f:
                            f.write(old_file.getbuffer())
                        with open(new_path, "wb") as f:
                            f.write(new_file.getbuffer())

                        # Initialize analyzer
                        analyzer = VersionComparisonAnalyzer(old_path, new_path)

                        # Process data
                        analyzer.merge_datasets()
                        analyzer.calculate_differences()

                        # Create results directory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_dir = os.path.join(
                            temp_dir, f"comparison_results_{timestamp}"
                        )
                        os.makedirs(results_dir, exist_ok=True)

                        # Generate reports and plots
                        version_report = analyzer.generate_summary_report()
                        with open(
                            os.path.join(results_dir, "version_comparison_report.md"),
                            "w",
                        ) as f:
                            f.write(version_report)

                        analyzer.plot_summary_changes()
                        plt.savefig(os.path.join(results_dir, "summary_changes.png"))
                        plt.close()

                        analyzer.generate_heatmap()
                        analyzer.export_detailed_statistics()
                        analyzer.generate_category_analysis()
                        analyzer.generate_table_heatmaps()
                        analyzer.generate_summary_csv()

                        # Create zip file
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(
                            zip_buffer, "w", zipfile.ZIP_DEFLATED
                        ) as zip_file:
                            for root, _, files in os.walk(results_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arc_name = os.path.relpath(file_path, results_dir)
                                    zip_file.write(file_path, arc_name)

                        st.success("Processing complete! Click below to download the analysis results.")
                        
                        # Offer download button
                        st.download_button(
                            label="Download Analysis Results",
                            data=zip_buffer.getvalue(),
                            file_name=f"comparison_results_{timestamp}.zip",
                            mime="application/zip",
                        )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
