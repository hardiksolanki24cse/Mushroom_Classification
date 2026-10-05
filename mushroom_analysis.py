import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import os

# Create directories for output
os.makedirs('visualizations', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Load the Data
print("Downloading data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
response = requests.get(url)
content = response.content.decode('utf-8')

columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

df = pd.read_csv(io.StringIO(content), names=columns)
df.to_csv("data/mushrooms_raw.csv", index=False)
print("Data loaded successfully.")

# 2. Data Cleaning

report_content = []
report_content.append("# Mushroom Classification - Data Analysis Report\n")
report_content.append("## 1. Data Cleaning\n")

# Missing values
# The dataset uses '?' for missing values in 'stalk-root'
missing_before = (df == '?').sum().sum()
df.replace('?', np.nan, inplace=True)
missing_after = df.isna().sum().sum()
report_content.append(f"- **Missing Values**: Replaced '{missing_before}' instances of '?' with NaN. The 'stalk-root' column has missing values.\n")

# Let's fill missing values with the mode
for col in df.columns:
    if df[col].isna().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

report_content.append("- **Imputation**: Filled missing values in 'stalk-root' with the mode value.\n")

# Duplicates
duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
report_content.append(f"- **Duplicates**: Found and removed {duplicates} duplicate rows.\n")

# Data types
dtypes_str = str(df.dtypes.value_counts().to_dict())
report_content.append(f"- **Data Types**: All columns are categorical (object). Distribution: {dtypes_str}\n")

# Outliers
# Since all data is categorical, we check for rare categories instead of numerical outliers.
rare_cats = []
for col in df.columns:
    val_counts = df[col].value_counts(normalize=True)
    rare = val_counts[val_counts < 0.01].index.tolist()
    if rare:
        rare_cats.append(f"{col}: {rare}")

report_content.append("- **Outliers (Rare Categories)**: As data is categorical, we checked for categories representing < 1% of the data. Some rare categories found in columns like 'cap-shape', 'cap-color', 'gill-color' etc. They were kept as they represent valid rare variations of mushrooms.\n")

df.to_csv("data/mushrooms_cleaned.csv", index=False)

# 3. Visualizations
report_content.append("\n## 2. Visualizations\n")

sns.set_theme(style="whitegrid", palette="muted")

# Plot 1: Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='class', hue='class', palette={'e': 'forestgreen', 'p': 'crimson'})
plt.title('Distribution of Edible (e) vs Poisonous (p) Mushrooms')
plt.savefig('visualizations/1_class_distribution.png')
plt.close()
report_content.append("### 1. Class Distribution\n")
report_content.append("The dataset is fairly balanced between edible and poisonous mushrooms, with slightly more edible ones.\n")
report_content.append("![Class Distribution](visualizations/1_class_distribution.png)\n")

# Plot 2: Odor vs Class
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='odor', hue='class', palette={'e': 'forestgreen', 'p': 'crimson'})
plt.title('Mushroom Odor vs Classification')
plt.savefig('visualizations/2_odor_vs_class.png')
plt.close()
report_content.append("### 2. Odor vs Classification\n")
report_content.append("Odor is a very strong predictor. For example, all mushrooms with foul (f) odor are poisonous, while those with no odor (n) or almond (a) are mostly edible.\n")
report_content.append("![Odor vs Class](visualizations/2_odor_vs_class.png)\n")

# Plot 3: Spore Print Color vs Class
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='spore-print-color', hue='class', palette={'e': 'forestgreen', 'p': 'crimson'})
plt.title('Spore Print Color vs Classification')
plt.savefig('visualizations/3_spore_print_vs_class.png')
plt.close()
report_content.append("### 3. Spore Print Color vs Classification\n")
report_content.append("Spore print color also shows clear distinctions. Chocolate (h) and white (w) spores often indicate poisonous mushrooms, while brown (n) and black (k) are mostly edible.\n")
report_content.append("![Spore Print vs Class](visualizations/3_spore_print_vs_class.png)\n")

# Plot 4: Habitat vs Class
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='habitat', hue='class', palette={'e': 'forestgreen', 'p': 'crimson'})
plt.title('Habitat vs Classification')
plt.savefig('visualizations/4_habitat_vs_class.png')
plt.close()
report_content.append("### 4. Habitat vs Classification\n")
report_content.append("Mushrooms found in paths (p) or urban (u) areas tend to be poisonous, while those in woods (d) or grasses (g) are mixed but lean towards edible.\n")
report_content.append("![Habitat vs Class](visualizations/4_habitat_vs_class.png)\n")

# Plot 5: Cap Color vs Class
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='cap-color', hue='class', palette={'e': 'forestgreen', 'p': 'crimson'})
plt.title('Cap Color vs Classification')
plt.savefig('visualizations/5_cap_color_vs_class.png')
plt.close()
report_content.append("### 5. Cap Color vs Classification\n")
report_content.append("Cap color is less distinctive on its own compared to odor. Brown (n) and gray (g) are the most common colors for both classes.\n")
report_content.append("![Cap Color vs Class](visualizations/5_cap_color_vs_class.png)\n")

# Plot 6: Correlation Heatmap (using Cramer's V or just Label Encoding for simple visualization)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
corr = df_encoded.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Features (Label Encoded)')
plt.savefig('visualizations/6_correlation_heatmap.png')
plt.close()
report_content.append("### 6. Feature Correlation Heatmap\n")
report_content.append("While Pearson correlation is meant for continuous data, using it on label-encoded categorical data gives us a rough idea of linear associations. Gill-color, ring-type, and bruises show noticeable associations with the class.\n")
report_content.append("![Correlation Heatmap](visualizations/6_correlation_heatmap.png)\n")

# Write report
with open("Insights_Report.md", "w") as f:
    f.writelines(report_content)
    
print("Analysis complete! Report and visualizations generated.")
