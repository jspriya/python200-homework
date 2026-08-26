import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from prefect import flow, task, get_run_logger

@task(retries=3, retry_delay_seconds=2)
def load_all_happiness_data(base_url, years):
    logger = get_run_logger()

    all_data = []

    for year in years:
        logger.info(f"Loading {year}")

        df = pd.read_csv(f"{base_url}{year}.csv", sep=";", decimal=",")

        # Standardize column names
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
        )
        # Add year column
        df["year"] = int(year)
        all_data.append(df)
    
    # Merge all years
    merged_df = pd.concat(all_data, ignore_index=True)

     # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save merged dataset
    output_file = os.path.join(output_dir, "merged_happiness.csv")
    merged_df.to_csv(output_file, index=False)

    logger.info(f"Shape: {merged_df.shape}")
    logger.info(f"Preview:\n{merged_df.head()}")

    return merged_df

@task
def descriptive_statistics(df: pd.DataFrame):
    logger = get_run_logger()

    # ----------------------------
    # Overall statistics
    # ----------------------------
    mean_score = df["happiness_score"].mean()
    median_score = df["happiness_score"].median()
    std_score = df["happiness_score"].std()

    logger.info("=== Overall Happiness Score Stats ===")
    logger.info(f"Mean: {mean_score}")
    logger.info(f"Median: {median_score}")
    logger.info(f"Std Dev: {std_score}")

    # ----------------------------
    # Group by year
    # ----------------------------
    year_mean = df.groupby("year")["happiness_score"].mean()

    logger.info("=== Mean Happiness by Year ===")
    logger.info(f"\n{year_mean}")

    # ----------------------------
    # Group by region
    # ----------------------------
    region_mean = df.groupby("regional_indicator")["happiness_score"].mean().sort_values(ascending=False)

    logger.info("=== Mean Happiness by Region ===")
    logger.info(f"\n{region_mean}")

    # Return everything for later tasks
    return {
        "overall_mean": mean_score,
        "overall_median": median_score,
        "overall_std": std_score,
        "year_mean": year_mean,
        "region_mean": region_mean
    }

@task
def create_visualizations(df):
    logger = get_run_logger()

    # Output directory (inside assignments_01)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # 1. Histogram
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["happiness_score"], bins=20)
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")

    histogram_path = os.path.join(output_dir, "happiness_histogram.png")
    plt.savefig(histogram_path)
    plt.close()

    logger.info(f"Saved histogram to {histogram_path}")
     
    # ----------------------------
    # 2. Boxplot by Year
    # ----------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="year", y="happiness_score")

    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")

    boxplot_path = os.path.join(output_dir, "happiness_by_year.png")
    plt.savefig(boxplot_path)
    plt.close()

    logger.info(f"Saved boxplot to {boxplot_path}")

    # ----------------------------
    # 3. GDP vs Happiness Scatter Plot
    # ----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(df["gdp_per_capita"], df["happiness_score"])

    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")

    scatter_path = os.path.join(output_dir, "gdp_vs_happiness.png")
    plt.savefig(scatter_path)
    plt.close()

    logger.info(f"Saved scatter plot to {scatter_path}")

    # ----------------------------
    # 4. Correlation Heatmap
    # ----------------------------
    plt.figure(figsize=(10, 8))

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=True)

    plt.title("Correlation Heatmap")

    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    logger.info(f"Saved heatmap to {heatmap_path}")

@task
def hypothesis_testing(df):
    logger = get_run_logger()

    # -----------------------------------
    # Test 1: 2019 vs 2020
    # -----------------------------------
    scores_2019 = df[df["year"] == 2019]["happiness_score"]
    scores_2020 = df[df["year"] == 2020]["happiness_score"]

    t_stat, p_value = ttest_ind(scores_2019, scores_2020, equal_var=False)

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info("===== 2019 vs 2020 Happiness Scores =====")
    logger.info(f"2019 Mean: {mean_2019:.3f}")
    logger.info(f"2020 Mean: {mean_2020:.3f}")
    logger.info(f"T-statistic: {t_stat:.4f}")
    logger.info(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        logger.info(
            "The difference in average happiness scores between 2019 and 2020 "
            "is statistically significant. It is unlikely that this difference "
            "occurred by random chance."
        )
    else:
        logger.info(
            "The difference in average happiness scores between 2019 and 2020 "
            "is not statistically significant. The observed difference could "
            "reasonably be due to random variation."
        )

    # -----------------------------------
    # Test 2: Western Europe vs Sub-Saharan Africa
    # -----------------------------------
    europe = df[df["regional_indicator"] == "Western Europe"]["happiness_score"]
    africa = df[df["regional_indicator"] == "Sub-Saharan Africa"]["happiness_score"]

    t_stat2, p_value2 = ttest_ind(europe, africa, equal_var=False)

    mean_europe = europe.mean()
    mean_africa = africa.mean()

    logger.info("===== Western Europe vs Sub-Saharan Africa =====")
    logger.info(f"Western Europe Mean: {mean_europe:.3f}")
    logger.info(f"Sub-Saharan Africa Mean: {mean_africa:.3f}")
    logger.info(f"T-statistic: {t_stat2:.4f}")
    logger.info(f"P-value: {p_value2:.4f}")

    if p_value2 < 0.05:
        logger.info(
            "Western Europe has a statistically significantly higher average "
            "happiness score than Sub-Saharan Africa in this dataset."
        )
    else:
        logger.info(
            "No statistically significant difference was found between the two regions."
        )

@task
def correlation_analysis(df: pd.DataFrame):
    logger = get_run_logger()

    logger.info("===== Correlation Analysis =====")

    # Select numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Explanatory variables (exclude target)
    explanatory_vars = [
        col for col in numeric_df.columns
        if col != "happiness_score"
    ]

    number_of_tests = len(explanatory_vars)
    adjusted_alpha = 0.05 / number_of_tests

    logger.info(f"Number of tests: {number_of_tests}")
    logger.info(f"Original alpha: 0.05")
    logger.info(f"Bonferroni-adjusted alpha: {adjusted_alpha:.6f}")

    for column in explanatory_vars:

        corr, p_value = pearsonr(
            numeric_df[column],
            numeric_df["happiness_score"]
        )

        significant_original = p_value < 0.05
        significant_adjusted = p_value < adjusted_alpha

        logger.info("----------------------------------------")
        logger.info(f"Variable: {column}")
        logger.info(f"Correlation: {corr:.4f}")
        logger.info(f"P-value: {p_value:.6f}")
        logger.info(
            f"Significant at α = 0.05: {significant_original}"
        )
        logger.info(
            f"Significant after Bonferroni: {significant_adjusted}"
        )
@task
def summary_report(df):
    logger = get_run_logger()

    # ----------------------------
    # Total countries and years
    # ----------------------------
    num_countries = df["country"].nunique()
    num_years = df["year"].nunique()

    logger.info(
        f"The merged dataset contains {num_countries} unique countries "
        f"across {num_years} years."
    )
    # ----------------------------
    # Top and bottom 3 regions
    # ----------------------------
    region_means = (
        df.groupby("regional_indicator")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
    )

    top3 = region_means.head(3)
    bottom3 = region_means.tail(3)

    logger.info(f"Top 3 regions by mean happiness:\n{top3}")
    logger.info(f"Bottom 3 regions by mean happiness:\n{bottom3}")
    
    # ----------------------------
    # 2019 vs 2020 t-test summary
    # ----------------------------
    from scipy.stats import ttest_ind

    scores_2019 = df[df["year"] == 2019]["happiness_score"]
    scores_2020 = df[df["year"] == 2020]["happiness_score"]

    t_stat, p_value = ttest_ind(
        scores_2019,
        scores_2020,
        equal_var=False
    )

    if p_value < 0.05:
        logger.info(
            "The average happiness score changed significantly between "
            "2019 and 2020. The observed difference is unlikely to be "
            "explained by random chance."
        )
    else:
        logger.info(
            "The average happiness score did not change significantly "
            "between 2019 and 2020."
        )
    # ----------------------------
    # Strongest correlation
    # ----------------------------
    numeric_df = df.select_dtypes(include="number")

    explanatory = [
        c for c in numeric_df.columns
        if c != "happiness_score"
    ]

    adjusted_alpha = 0.05 / len(explanatory)

    best_variable = None
    best_corr = 0
    best_p = None

    for column in explanatory:

        corr, p = pearsonr(
            numeric_df[column],
            numeric_df["happiness_score"]
        )

        if p < adjusted_alpha:

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_variable = column
                best_p = p

    if best_variable:
        logger.info(
            f"The variable most strongly correlated with happiness score "
            f"after Bonferroni correction is '{best_variable}' "
            f"(correlation = {best_corr:.3f}, p = {best_p:.6f})."
        )
    else:
        logger.info(
            "No variables remained statistically significant after "
            "Bonferroni correction."
        )

@flow
def happiness_pipeline():
    base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_"

    years = ["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]

    df = load_all_happiness_data(base_url, years)

    stats = descriptive_statistics(df)

    create_visualizations(df)

    hypothesis_testing(df)

    correlation_analysis(df)

    summary_report(df)

    #return stats

if __name__ == "__main__":
    happiness_pipeline()