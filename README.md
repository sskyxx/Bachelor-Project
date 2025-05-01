# Bachelor Project - "Examining Nutritional Quality of Meals On a Swiss Digital Nutrition Cohort"


## Introduction
This repository implements a set of tools and analyses for assessing the **nutrient balance concept** of individual food and composite meals. We compute three indices :
- **QI** (Qualifying Index) : a measure based on qualifying/good micronutrients
- **DI** (Disqualifying Index) : a measure based on disqualifying/bad micronutrients
- **NB** (Nutrient Balance) : an overall balance score (%)

Using these results, we analyse :
- How food group and meal-type affect nutrient quality
- Time-wise daily pattern
- Which micronutrients most inflence QI/DI


---

## Features
- Compute nutrient quality indices (QI, DI and NB) for meals
- Clean data my removing outliers
- Visualize
    - Distributions by food group
    - Weekday vs. weekend meal type
    - Composite meal
- Rank micronutrients impacts on nutrient quality indices


---
## Setup
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

1. Clone this repo
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
## Usage
### Data preparation
Run `analysis/nutrient_metric.ipynb`to :
- Load raw data food
- Compute QI, DI and NB per item

### Running analyses
Run the notebooks in `analysis/` :
- `outliers.ipynb` 
- `food_group_analysis.ipynb`
- `food_item_composed.ipynb`
- `composite_meals.ipynb`

---

## Scripts & modules

All core utility are in `scripts/functions.py`

