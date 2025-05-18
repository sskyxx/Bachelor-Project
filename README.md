# Nutrient Quality Analysis for Swiss Digital Nutrition Cohort

## 📋 Project Overview
This project implements a **comprehensive framework** for assessing the **nutrient balance** of individual food items and composite meals, using three innovative indices:
- **QI (Qualifying Index)**: Measures beneficial micronutrients.
- **DI (Disqualifying Index)**: Measures less desirable micronutrients.
- **NB (Nutrient Balance)**: Overall nutrient balance score (%).

### Key Analyses:
- Nutrient quality across food groups and meal types.
- Temporal patterns in meal quality.
- Micronutrient impact on QI/DI.

---

## 🌟 Features
- **Quality Index Calculation**: QI, DI, NB computation.
- **Data Cleaning**: Outlier detection and removal.
- **Visualizations**:
  - Nutrient distribution by food group.
  - Weekday vs. weekend analysis.
  - Composite meal assessment.
- **Ranking**: Identify top micronutrient contributors.


---
## 🚀 Setup
```yaml
name : nutrient-env
channels :
- defaults
dependencies :
- python=3.10
- pandas
- numpy
- matplotlib
- seaborn
- ipywidgets
- scipy
- scikit-learn
```

1. Clone the repository
```bash
git clone https://github.com/sskyxx/Bachelor-Project.git
cd Bachelor-Project
```
2. Create & activate conda environment 
```
conda env create -f environment.yml
conda activate nutrient-env
```
3. Install dependencies
```bash
pip install -r requirements.txt
``` 
---
## 📊 Usage
### Data preparation
Run `analysis/nutrient_metric.ipynb`to :
- Load raw data food
- Compute QI, DI and NB per item

### Running analyses
Run the notebooks in `analysis/` :
- `outliers.ipynb` : Data cleaning and outlier removal
- `food_group_analysis.ipynb` : Food group analysis
- `composite_meals.ipynb` : Composite meal analysis
- `impact_on_indices.ipynb` : Impact of micronutrient in QI/DI
- `statistical_tests.ipynb`: Statistical tests

---

## 🛠️ Scripts & modules

All core utility are in `scripts/functions.py`

