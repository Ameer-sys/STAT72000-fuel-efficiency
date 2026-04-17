import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway, t
import statsmodels.api as sm

# making sure output folders exist before saving anything
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# 1. LOAD THE DATA
# reading the csv file
df = pd.read_csv("data/raw/cars.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nDataset shape before cleaning:")
print(df.shape)

# 2. CLEAN THE DATA
# checking missing values first
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# removing duplicate rows
df = df.drop_duplicates()

# removing rows with missing values
df = df.dropna()

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# saving cleaned version so we can include it if needed
df.to_csv("outputs/tables/cleaned_cars.csv", index=False)

# 3. DESCRIPTIVE STATISTICS
# these are the main numeric columns we care about
numeric_cols = ["city_mpg", "highway_mpg", "combination_mpg", "cylinders", "displacement"]

print("\nDescriptive statistics:")
print(df[numeric_cols].describe())

# creating a nicer summary table with the required stats
summary_table = pd.DataFrame({
    "Mean": df[numeric_cols].mean(),
    "Median": df[numeric_cols].median(),
    "Mode": df[numeric_cols].mode().iloc[0],
    "Variance": df[numeric_cols].var(),
    "Standard Deviation": df[numeric_cols].std(),
    "Minimum": df[numeric_cols].min(),
    "Maximum": df[numeric_cols].max()
})

# adding IQR too since it is useful for spread and outliers
summary_table["Q1"] = df[numeric_cols].quantile(0.25)
summary_table["Q3"] = df[numeric_cols].quantile(0.75)
summary_table["IQR"] = summary_table["Q3"] - summary_table["Q1"]

print("\nSummary table:")
print(summary_table)

summary_table.to_csv("outputs/tables/summary_statistics.csv")

# 4. FREQUENCY TABLES
# class frequency
class_freq = df["class"].value_counts().to_frame(name="Count")
class_freq["Percent"] = (df["class"].value_counts(normalize=True) * 100).round(2)
class_freq.to_csv("outputs/tables/class_frequency.csv")

# fuel type frequency
fuel_freq = df["fuel_type"].value_counts().to_frame(name="Count")
fuel_freq["Percent"] = (df["fuel_type"].value_counts(normalize=True) * 100).round(2)
fuel_freq.to_csv("outputs/tables/fuel_type_frequency.csv")

# transmission frequency
trans_freq = df["transmission"].value_counts().to_frame(name="Count")
trans_freq["Percent"] = (df["transmission"].value_counts(normalize=True) * 100).round(2)
trans_freq.to_csv("outputs/tables/transmission_frequency.csv")

# 5. VISUALIZATIONS
# histogram for city mpg
plt.figure(figsize=(8, 5))
df["city_mpg"].hist(bins=20)
plt.title("Histogram of City MPG")
plt.xlabel("City MPG")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/hist_city_mpg.png")
plt.close()

# histogram for highway mpg
plt.figure(figsize=(8, 5))
df["highway_mpg"].hist(bins=20)
plt.title("Histogram of Highway MPG")
plt.xlabel("Highway MPG")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/hist_highway_mpg.png")
plt.close()

# boxplot for city mpg
plt.figure(figsize=(6, 5))
plt.boxplot(df["city_mpg"])
plt.title("Boxplot of City MPG")
plt.ylabel("City MPG")
plt.tight_layout()
plt.savefig("outputs/figures/boxplot_city_mpg.png")
plt.close()

# bar chart for vehicle class
plt.figure(figsize=(10, 5))
df["class"].value_counts().plot(kind="bar")
plt.title("Vehicle Class Distribution")
plt.xlabel("Vehicle Class")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/figures/bar_vehicle_class.png")
plt.close()

# scatterplot for displacement vs city mpg
plt.figure(figsize=(8, 5))
plt.scatter(df["displacement"], df["city_mpg"])
plt.title("Displacement vs City MPG")
plt.xlabel("Displacement")
plt.ylabel("City MPG")
plt.tight_layout()
plt.savefig("outputs/figures/scatter_displacement_city_mpg.png")
plt.close()

# line chart for average city mpg by year
year_avg = df.groupby("year")["city_mpg"].mean()
plt.figure(figsize=(8, 5))
plt.plot(year_avg.index, year_avg.values, marker="o")
plt.title("Average City MPG by Year")
plt.xlabel("Year")
plt.ylabel("Average City MPG")
plt.tight_layout()
plt.savefig("outputs/figures/line_city_mpg_by_year.png")
plt.close()

# 6. CORRELATION MATRIX
corr_matrix = df[numeric_cols].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

corr_matrix.to_csv("outputs/tables/correlation_matrix.csv")

# simple heatmap using matplotlib only
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, interpolation="nearest", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/figures/correlation_matrix.png")
plt.close()

# 7. PEARSON CORRELATION TEST
# checking relationship between displacement and city mpg
r_value, p_value = pearsonr(df["displacement"], df["city_mpg"])

print("\nPearson Correlation Test: displacement vs city_mpg")
print("Correlation coefficient:", r_value)
print("P-value:", p_value)

pearson_results = pd.DataFrame({
    "Variable 1": ["displacement"],
    "Variable 2": ["city_mpg"],
    "Correlation Coefficient": [r_value],
    "P-value": [p_value]
})

pearson_results.to_csv("outputs/tables/pearson_correlation_results.csv", index=False)

# 8. 95% CONFIDENCE INTERVAL FOR CITY MPG
# confidence interval for the mean city mpg
n = len(df["city_mpg"])
mean_city = df["city_mpg"].mean()
std_city = df["city_mpg"].std()
standard_error = std_city / (n ** 0.5)

confidence_level = 0.95
t_critical = t.ppf((1 + confidence_level) / 2, df=n - 1)

margin_of_error = t_critical * standard_error
lower_bound = mean_city - margin_of_error
upper_bound = mean_city + margin_of_error

print("\n95% Confidence Interval for mean city MPG:")
print("Mean:", mean_city)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

ci_table = pd.DataFrame({
    "Mean City MPG": [mean_city],
    "Lower Bound": [lower_bound],
    "Upper Bound": [upper_bound]
})

ci_table.to_csv("outputs/tables/confidence_interval_city_mpg.csv", index=False)

# 9. ANOVA TEST
# comparing city mpg across vehicle classes
groups = [group["city_mpg"].values for name, group in df.groupby("class") if len(group) > 1]

if len(groups) > 1:
    f_stat, anova_p = f_oneway(*groups)

    print("\nANOVA Test: city_mpg by class")
    print("F-statistic:", f_stat)
    print("P-value:", anova_p)

    anova_results = pd.DataFrame({
        "Test": ["ANOVA"],
        "Dependent Variable": ["city_mpg"],
        "Grouping Variable": ["class"],
        "F-statistic": [f_stat],
        "P-value": [anova_p]
    })

    anova_results.to_csv("outputs/tables/anova_results.csv", index=False)

# 10. SIMPLE REGRESSION MODEL
# using a simple regression model to predict city mpg
# predictors chosen because they are directly related to engine size/performance
X = df[["displacement", "cylinders", "year"]]
y = df["city_mpg"]

# adding constant for intercept
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\nRegression summary:")
print(model.summary())

# saving regression coefficients
regression_table = pd.DataFrame({
    "Coefficient": model.params,
    "P-value": model.pvalues
})

regression_table.to_csv("outputs/tables/regression_results.csv")

# residual plot
plt.figure(figsize=(8, 5))
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0)
plt.title("Residual Plot")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("outputs/figures/residual_plot.png")
plt.close()

# 11. FINAL MESSAGE
print("\nDone.")
print("Check outputs/figures for charts.")
print("Check outputs/tables for summary tables and test results.")