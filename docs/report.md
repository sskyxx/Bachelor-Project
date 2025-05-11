# Examining Nutritional Quality of Meals On a Swiss Digital Nutrition Cohort - Report

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


### **Nutrient Index Computation**

#### **1. Nutrient Ratios**  
For each nutrient, we compute:  
$$
\text{ratio} = \frac{\text{amount} \times \text{conversion\_factor}}{\text{DRI\_target}}
$$

#### **2. Energy Scaling**  
Normalize ratios to a 2000 kcal intake:  
$$
\text{scaled\_ratio} = \text{ratio} \times \left(\frac{2000}{\text{energy\_kcal\_eaten}}\right)
$$

---

### **Composite Indices**

#### **Qualifying Index (QI)**  
Measures nutrient density relative to energy content:  
$$
QI = \frac{E_d}{E_p} \cdot \frac{\sum_{j=1}^{N_q} \frac{a_{qj}}{r_{qj}}}{N_q}
$$

**Variables**:  
- $ E_d $ : Daily energy needs (default = 2000 kcal)
- $ E_p $: Energy in the food/meal (kcal)  
- $ a_{qj} $ : Amount of qualifying nutrient $ j $ (g, mg, or µg)
- $ r_{qj} $ : DRI for qualifying nutrient $ j $ 
- $ N_q $ : Number of qualifying nutrients

**Interpretation**:  
- $ QI > 1 $: Nutrient-dense food
- $ QI < 1 $ : Energy-dense food



#### **Disqualifying Index (DI)**  
Quantifies excess of harmful nutrients:  

$$
DI = \frac{E_d}{E_p} \cdot \frac{\sum_{j=1}^{N_d} \frac{a_{dj}}{r_{dj}}}{N_d}
$$

**Variables**:  
- $ a_{dj} $ : Amount of disqualifying nutrient $ j $ (g or mg)
- $ r_{dj} $ : MRV (Maximal Reference Value) for nutrient $ j $
- $ N_d $ : Number of disqualifying nutrients (6: fat, saturated fat, trans fat, cholesterol, sodium, sugar)

**Interpretation**:  
- $ DI < 1 $ : Low in harmful nutrients
- $ DI > 1 $ : Excessive intake


#### **Nutrient Balance (NB)**  
Percentage of qualifying nutrients meeting 100% of DRI:  

$$
NB(\%) = \frac{\sum_{j=1}^{N_q} q_j'}{N_q} \times 100
$$  
where $ q_j' = \min\left(\frac{a_{qj}}{r_{qj}}, 1\right) $ (truncated at 1.0)

**Interpretation**:  
- $ NB = 100\% $ : All qualifying nutrients are fully satisfied

--- 

### **Nutrient Index Sensitivity**

#### **Impact on QI**

Low $ qi < 1 $ indicates a nutrient deficency, meaning the food provides less of that nutrient (pere 2000 kcal) than the DRI. For exemple, if a food item as a $ qi=0.2$ for vitamin c, it only provides 20% of the DRI of this micronutrient per 2000 kcal. Because **QI** is an average of all $ qi $ values, improving very low $qi$ has a larger effect on the mean than improving already high $qi$. The **QI** is therefore most sensitive to the *worst-performing nutrients* in a diet.

#### **Impact on DI**

High $ di > 1$ signals excessive intake of disqualifying nutrients relative to their MRV. For instance, if a food as a $di = 2.5$ for sodium, it's provides 250% of the recommended maximum. Because **DI** is the mean of all $di$ values, reducing the highest $di$ nutrient lowers **DI** more than reducing nutrients that are already low. **DI** is consequently most responsive to the *most excessive problematic nutrients*. 

#### **Impact on NB**

The **NB** score equally weights all qualifying nutrients with $qi < 1$ (truncated at 1.0). Unlike QI, **NB** doesn't prioritize the most deficient nutrients. It rewards balanced improvement across all insufficient nutrient. Therefore correcting either $qi < 1$ to $qi > 1$ improves NB by the same amount. In a meal composition, pairing foods with complentary nutrient profiles, avoiding redundants food item, helps imprving the overall **NB** score.


--- 
## Statistical analyses


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
