# Mushroom Classification Analysis

This project analyzes the **Mushroom Classification** dataset from the UCI Machine Learning Repository to determine which features are most indicative of whether a mushroom is edible or poisonous. 

## 🍄 Project Overview

The dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This project contains a Python script that automates the downloading, cleaning, analyzing, and visualization of this dataset.

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hardiksolanki24cse/Mushroom_Classification.git
   cd Mushroom_Classification
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis script:**
   ```bash
   python mushroom_analysis.py
   ```

## 📊 What the Code Does (`mushroom_analysis.py`)

The python script performs the following operations:
1. **Data Loading**: Fetches the raw dataset directly from the UCI Machine Learning Repository URL.
2. **Data Cleaning**:
   - **Missing Values**: Identifies and handles missing values (replaces `'?'` in `stalk-root` with the mode).
   - **Duplicates**: Finds and removes duplicate rows.
   - **Data Types**: Confirms all features are categorical text.
   - **Outliers**: Analyzes categorical outliers (rare categories with < 1% occurrence) and keeps them as they represent valid rare variations in nature.
3. **Data Visualization**: Generates 6 distinct visual plots exploring relationships between features (like Odor, Spore Print Color, and Cap Color) and mushroom edibility. Images are saved into the `visualizations/` directory.
4. **Insights Generation**: Automatically generates a detailed `Insights_Report.md` summarizing the findings from the visualizations and cleaning processes.

## 📁 Repository Structure

- `mushroom_analysis.py`: The main Python script that runs the entire analysis.
- `requirements.txt`: Python package dependencies (pandas, seaborn, matplotlib, scikit-learn, etc.).
- `data/`: Folder containing both the raw downloaded data (`mushrooms_raw.csv`) and the cleaned dataset (`mushrooms_cleaned.csv`).
- `visualizations/`: Contains 6 generated plots in PNG format (Class Distribution, Odor vs Class, etc.).
- `Insights_Report.md`: A generated report summarizing data cleaning steps and insights derived from the visualizations.
