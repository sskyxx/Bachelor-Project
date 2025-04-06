import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Circle
import os
import pickle
from scipy.stats import mannwhitneyu, ttest_ind, kruskal, f_oneway
from statannotations.Annotator import Annotator
from itertools import combinations
from config import nutrient_info, conversion_factors

def calculate_ratios(df, nutrient_cols):
    df = df.copy()

    for nutr in nutrient_cols:
        conversion_factor = conversion_factors[nutrient_info[nutr]['unit']]

        df[nutr + '_ratio'] = df[nutr] * conversion_factor / nutrient_info[nutr]['target']

    return df


def scale(df, nutrient_cols, scaling_factor=2000):
    df = df.copy()

    for nutr in nutrient_cols:

        df[nutr + "_ratio_scaled"] = df[nutr + '_ratio']* (scaling_factor / df["energy_kcal_eaten"])
     

    return df


def compute_index(row, nutrient_cols, scaling_factor=2000) :

    index = 0
    ratio_sum = 0

    for nutr in nutrient_cols :
        ratio_sum += row[nutr + '_ratio']

    index = (scaling_factor / row['energy_kcal_eaten']) * (ratio_sum / len(nutrient_cols))
    return index


def compute_nb(row, nutrient_cols, scaling_factor=2000) :
    truncated_ratios = []
    for nutr in nutrient_cols:
        ratio =   row[nutr + '_ratio_scaled']
        if ratio > 1.0 :
            ratio = 1
        truncated_ratios.append(ratio)

    nb_value =  (sum(truncated_ratios) / len(nutrient_cols)) * 100
    return nb_value


def compute_qi_excluding(row, nutrient_list, exclude=None, scaling_factor=2000):
    if exclude is not None:
        new_list = [nutr for nutr in nutrient_list if nutr != exclude]
    else:
        new_list = nutrient_list
    

    return compute_index(row, new_list, scaling_factor=scaling_factor)



def compare_qi_excluding_nutrient(df, nutrient_to_exclude, qualifying_nutrients, new_col_name=None,scaling_factor=2000):


    if new_col_name is None:
        new_col_name = f"QI_excl_{nutrient_to_exclude}"


    df[new_col_name] = df.apply(lambda row: compute_qi_excluding(row, qualifying_nutrients, exclude=nutrient_to_exclude, scaling_factor=scaling_factor), axis=1)

 
    df_plot = df.drop_duplicates('combined_name').copy()
    df_plot = df_plot[['combined_name', 'QI', new_col_name]]

    labels = df_plot['combined_name'].tolist()
    x = np.arange(len(labels))
    width = 0.35


    plt.figure(figsize=(20, 8))
    plt.bar(x - width/2, df_plot['QI'], width, label='QI (incl. ' + nutrient_to_exclude + ')', color='skyblue')
    plt.bar(x + width/2, df_plot[new_col_name], width, label=f"QI (excl. {nutrient_to_exclude})", color='deeppink')

    plt.xlabel('Food Item')
    plt.ylabel('QI Value')
    plt.title(f"Comparison of QI with and without {nutrient_to_exclude}")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    df_plot['QI_diff'] = df_plot['QI'] - df_plot[new_col_name]
    df_plot['QI_pct_change'] = (df_plot['QI_diff'] / df_plot['QI']) * 100

    return df_plot


def compute_qi_excluding_multiple(row, nutrient_list, exclude_list, scaling_factor=2000):

    new_list = nutrient_list.copy()  

    for nutr in exclude_list:
        if nutr in new_list:
            new_list.remove(nutr)

    return compute_index(row, new_list, scaling_factor=scaling_factor)


def plot_nutrient_contributions_with_qi(row, nutrient_cols, exclude_list=None, scaling_factor=2000):

    if exclude_list is not None:
        included_nutrients = [nutr for nutr in nutrient_cols if nutr not in exclude_list]
    else:
        included_nutrients = nutrient_cols

    qi_value = compute_index(row, included_nutrients, scaling_factor=scaling_factor)
    
    ratio_cols = [nutr + '_ratio_scaled' for nutr in included_nutrients]

    values = row[ratio_cols]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(ratio_cols, values, color='skyblue', edgecolor='skyblue')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.05 * max(values),
                 f"{height:.2f}", ha='center', va='bottom', fontsize=8)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Scaled Nutrient Ratio")
    
 
    if exclude_list is None:
        title_str = f"Nutrient Contributions for {row['combined_name']} \n(QI = {qi_value:.2f})\n"
    else:
        title_str = (f"Nutrient Contributions for {row['combined_name']} \n(Excluding {', '.join(exclude_list)})\n"f"QI = {qi_value:.2f}\n")
    
    plt.title(title_str)
    plt.tight_layout()
    plt.show()


def compute_nb_excluding(row, nutrient_list, exclude=None, scaling_factor=2000):

    if exclude is not None:
        new_list = [nutr for nutr in nutrient_list if nutr != exclude]
    else:
        new_list = nutrient_list
    return compute_nb(row, new_list, scaling_factor=scaling_factor)
    

def compare_nb_excluding_nutrient(df, nutrient_to_exclude, qualifying_nutrients, new_col_name=None, scaling_factor=2000):
  
    if new_col_name is None:
        new_col_name = f"NB_excl_{nutrient_to_exclude}"

    df['NB'] = df.apply(lambda row: compute_nb(row, qualifying_nutrients, scaling_factor=scaling_factor), axis=1)

    df[new_col_name] = df.apply(lambda row: compute_nb_excluding(row, qualifying_nutrients, exclude=nutrient_to_exclude, scaling_factor=scaling_factor), axis=1)
    
    df_plot = df.drop_duplicates('combined_name').copy()
    df_plot = df_plot[['combined_name', 'NB', new_col_name]]
    
    labels = df_plot['combined_name'].tolist()
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(16, 8))
    plt.bar(x - width/2, df_plot['NB'], width, label='NB (incl. all)', color='skyblue')
    plt.bar(x + width/2, df_plot[new_col_name], width, label=f"NB (excl. {nutrient_to_exclude})", color='deeppink')
    plt.xlabel('Food Item')
    plt.ylabel('NB Value (%)')
    plt.title(f"Comparison of NB with and without {nutrient_to_exclude}")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    df_plot['NB_diff'] = df_plot['NB'] - df_plot[new_col_name]
    df_plot['NB_pct_change'] = (df_plot['NB_diff'] / df_plot['NB']) * 100
    return df_plot


def compute_nb_excluding_multiple(row, nutrient_list, exclude_list, scaling_factor=2000):
    new_list = nutrient_list.copy()  
    for nutr in exclude_list:
        if nutr in new_list:
            new_list.remove(nutr)
    return compute_nb(row, new_list, scaling_factor=scaling_factor)

def compare_di_excluding_nutrient(df, nutrient_to_exclude, disqualifying_nutrients, new_col_name=None, scaling_factor=2000):

    if new_col_name is None:
        new_col_name = f"DI_excl_{nutrient_to_exclude}"
    df[new_col_name] = df.apply(lambda row: compute_qi_excluding(row, disqualifying_nutrients, exclude=nutrient_to_exclude, scaling_factor=scaling_factor), axis=1)
    
    df_plot = df.drop_duplicates('combined_name').copy()
    df_plot = df_plot[['combined_name', 'DI', new_col_name]]
    
    labels = df_plot['combined_name'].tolist()
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(16, 8))
    plt.bar(x - width/2, df_plot['DI'], width, label='DI (incl. all)', color='skyblue')
    plt.bar(x + width/2, df_plot[new_col_name], width, label=f"DI (excl. {nutrient_to_exclude})", color='deeppink')
    
    plt.xlabel('Food Item')
    plt.ylabel('DI Value')
    plt.title(f"Comparison of DI with and without {nutrient_to_exclude}")
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    df_plot['DI_diff'] = df_plot['DI'] - df_plot[new_col_name]
    df_plot['DI_pct_change'] = (df_plot['DI_diff'] / df_plot['DI']) * 100
    
    return df_plot



