
# Correlation Matrices and ANOVA Assignment
# Based on the coding examples from class

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Folder for the results and visualizations
os.makedirs('analysis_results', exist_ok=True)


# PART 1: CORRELATION MATRIX - FLIGHT DELAYS

file = './Flight_Delays_2018.csv'
df = pd.read_csv(file)

# Select all numeric columns
numeric_columns = df.select_dtypes(include='number').columns

corr_matrix = df[numeric_columns].corr(
    numeric_only=True
).round(2)

print('\nFLIGHT DELAYS CORRELATION MATRIX')
print(corr_matrix)

corr_matrix.to_csv(
    './analysis_results/flight_correlation_matrix.csv'
)

plt.figure(figsize=(15, 12))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    annot_kws={'size': 6}
)

plt.title('Flight Delays Correlation Matrix')
plt.tight_layout()
plt.savefig('./analysis_results/flight_correlation.png')
plt.close()


# PART 2: CORRELATION MATRIX - COFFEE

file = './coffee.csv'
df = pd.read_csv(file)

numeric_columns = df.select_dtypes(include='number').columns

corr_matrix = df[numeric_columns].corr(
    numeric_only=True
).round(2)

print('\nCOFFEE CORRELATION MATRIX')
print(corr_matrix)

corr_matrix.to_csv(
    './analysis_results/coffee_correlation_matrix.csv'
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    annot_kws={'size': 7}
)

plt.title('Coffee Correlation Matrix')
plt.tight_layout()
plt.savefig('./analysis_results/coffee_correlation.png')
plt.close()


# PART 3: ANOVA - NO SIGNIFICANT DIFFERENCE

file = './coffee.csv'
df = pd.read_csv(file)

# Select three countries
query = "Location_Country in ['Costa Rica', 'Tanzania, United Republic Of', 'Thailand']"

df = df.query(query).dropna(
    subset=['Data_Scores_Total']
)

print('\nANOVA 1: NO SIGNIFICANT DIFFERENCE')

print(
    df.groupby('Location_Country')['Data_Scores_Total']
    .agg(['count', 'mean', 'std'])
)

# Create ANOVA model
model = sm_api.ols(
    'Data_Scores_Total ~ C(Location_Country)',
    data=df
).fit()

# Create ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

anova_table.to_csv(
    './analysis_results/anova_not_significant.csv'
)

# Check statistical significance
p_value = anova_table.loc[
    'C(Location_Country)', 'PR(>F)'
]

print('P-value:', p_value)
print('Significant difference:', p_value < 0.05)

# Create boxplot
plt.figure(figsize=(9, 5))

df.boxplot(
    column='Data_Scores_Total',
    by='Location_Country'
)

plt.title('ANOVA 1: Countries Without Significant Difference')
plt.suptitle('')
plt.xlabel('Country')
plt.ylabel('Coffee Total Score')
plt.tight_layout()

plt.savefig(
    './analysis_results/anova_not_significant_boxplot.png'
)

plt.close('all')


# PART 4: ANOVA - SIGNIFICANT DIFFERENCE

file = './coffee.csv'
df = pd.read_csv(file)

# Select three different countries
query = "Location_Country in ['United States', 'Brazil', 'Honduras']"

df = df.query(query).dropna(
    subset=['Data_Scores_Total']
)

print('\nANOVA 2: SIGNIFICANT DIFFERENCE')

print(
    df.groupby('Location_Country')['Data_Scores_Total']
    .agg(['count', 'mean', 'std'])
)

# Create ANOVA model
model = sm_api.ols(
    'Data_Scores_Total ~ C(Location_Country)',
    data=df
).fit()

# Create ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

anova_table.to_csv(
    './analysis_results/anova_significant.csv'
)

# Check statistical significance
p_value = anova_table.loc[
    'C(Location_Country)', 'PR(>F)'
]

print('P-value:', p_value)
print('Significant difference:', p_value < 0.05)

# Create boxplot
plt.figure(figsize=(9, 5))

df.boxplot(
    column='Data_Scores_Total',
    by='Location_Country'
)

plt.title('ANOVA 2: Countries With Significant Difference')
plt.suptitle('')
plt.xlabel('Country')
plt.ylabel('Coffee Total Score')
plt.tight_layout()

plt.savefig(
    './analysis_results/anova_significant_boxplot.png'
)

plt.close('all')

print('\nAssignment completed. Results saved in analysis_results.')
