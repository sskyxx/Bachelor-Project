# Examining Nutritional Quality of Meals On a Swiss Digital Nutrition Cohort

## Introduction
### Context & Motivation

Micronutrient adequacy is a cornerstone of dietary quality yet large-scale nutrition studies focus primarily on energy intake or macronutrient composition. In realty, under- or over- consumption of specific micronutrients (such as vitamins or minerals) can have effects on health and may vary systematically by food choice intake, meal timing and day of week.

In research of the diet quality, we impletent three complementary indices for each food item and composite meal :

1. **Qualifying Index QI** : Ratio of each qualifying nutrient contained in 2000 kcal of a given food relative to its Dietary Reference Intake (DRI) value
2. **Disqualifying Index DI** : Ratio of each disqualifying nutrient contained in 2000 kcal of a given food relative to its Dietary Reference Intake (DRI) value
3. **Nutrient Balance NB** : Score of the extent to which a food, meal or diet can satisfy the daily requirements for all qualifying nutrient in a sample containing 2000 kcal


### Aims & Questions
- How do nutrient quality indices (QI, DI, NB) vary across :
    - Different food groups?
    - Meal types (breakfast, lunch, snack, dinner)?
- What temporal patterns exist in nutritional quality :
    - Daily meal patterns?
    - Weekday vs. weekend differences?
- Which individual micronutrients most significantly impact :
    - Qualifying Index (QI)?
    - Disqualifying Index (DI)?

###
---

## Methods 

### Index computation
1. Nutrient Ratios :
    - For each nutrient, compute `(amount x conversion_factor) / DRI_target`
2. Energy Scaling :
    - Normalize to 2000 kcal intake : `ratio x (2000 / energy_kcal_eaten)`
3. Composite Indices :
    - **QI** : mean of scaled qualifying nutrient ratios
    - **DI** : mean of scaled disqualifying nuetrient ratios
    - **NB** : truncated mean of qualifying ratios (%)


### Statistical analyses
---


## Results
1. Food group distributions

2. Composite meals over time

3. Nutrient drivers
---

## Discussion
### Main findings

### Implications
- Provides quantitave framework for dietary quality assessment
- Enables targeted nutritional interventions
- Reveals temporal patterns in eating habits

### Limitations
- Dependene on accurate nutrient reporting
- Fixed DRIs may not account for individual variability
- Amount of `energy_kcal_eaten`taken in account 

### Future Directions
- Integration with clinical outcomes
- Personalized nutrient target


## Conclusion
