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
from matplotlib.lines import Line2D
import itertools
from statsmodels.stats.multitest import multipletests
from IPython.display import display

def calculate_ratios(df, nutrient_cols):
    """
    Compute for each nutrient the ratio = (amount * unit conversion) / target
    Return a new DataFrame w/ columns '{nutrient}_ratio'
    """
    df = df.copy()

    for nutr in nutrient_cols:
        conversion_factor = conversion_factors[nutrient_info[nutr]['unit']]

        df[nutr + '_ratio'] = df[nutr] * conversion_factor / nutrient_info[nutr]['target']

    return df


def scale(df, nutrient_cols, scaling_factor=2000):
    """
    Scale each '{nutrient}_ratio' by (scaling_factor / energy_kcal_eaten)
    Return a DataFrame w/ columns '{nutrient}_ratio_scaled'
    """
    df = df.copy()

    for nutr in nutrient_cols:

        df[nutr + "_ratio_scaled"] = df[nutr + '_ratio']* (scaling_factor / df["energy_kcal_eaten"])
     

    return df


def compute_index(row, nutrient_cols, scaling_factor=2000) :
    """
    Compute index (QI or DI)
    """

    index = 0
    ratio_sum = 0

    for nutr in nutrient_cols :
        ratio_sum += row[nutr + '_ratio']

    index = (scaling_factor / row['energy_kcal_eaten']) * (ratio_sum / len(nutrient_cols))
    return index


def compute_nb(row, nutrient_cols, scaling_factor=2000) :
    """
    Compute Nutrient Balance (NB)
    """

    truncated_ratios = []
    for nutr in nutrient_cols:
        ratio =   row[nutr + '_ratio_scaled']
        if ratio > 1.0 :
            ratio = 1
        truncated_ratios.append(ratio)

    nb_value =  (sum(truncated_ratios) / len(nutrient_cols)) * 100
    return nb_value

def filter_implausible_nutrients(df, nutrient_info, conversion_factors, max_dri_multiplier=3):
    """
    Drops rows where nutrient value exceed :
    - UL (if defined) or
    - max_dri_multipplied x DRI (if UL is None )
    
    Parameters
    ----------
    df : pd.DataFrame
    nutrient_info : dict
    conversion_factors : dict
    
    Returns
    -------
    pd.DataFrame
    """
    mask = pd.Series(True, index=df.index)       

    for nutr, info in nutrient_info.items():     
        if nutr not in df.columns:
            # skup if not inside the DataFrame
            print(f"Skipping {nutr} (not in DataFrame)") 
            continue

        factor = conversion_factors[info['unit']]
        converted = df[nutr] * factor

        # Case 1 : UL exist 
        ul = info['UL']

        print(f"\nNutrient: {nutr}")  # Debug: Show nutrient being processed
        print(f"Raw max: {df[nutr].max()} {info['unit']}")  # Debug: Show raw max value
        print(f"Converted max: {converted.max()} {info['unit']}")  # Debug: Show converted max
        print(f"UL: {ul}")  # Debug: Show UL

        if ul is not None:
            print(f"Rows exceeding UL: {(converted > ul).sum()}")  # Debug: Count violations
            mask &= (converted <= ul) | converted.isna()

        # Case 2 : no UL
        else:
            dri = info['target']
            max_plausible = dri * max_dri_multiplier
            print(f"Rows exceeding {max_dri_multiplier}x DRI: {(converted > max_plausible).sum()}")  # Debug
            mask &= (converted <= max_plausible) | converted.isna()
    print(f"\nTotal rows dropped: {len(df) - mask.sum()}")  # Debug: Total rows removed
    return df.loc[mask]



def compute_qi_excluding(row, nutrient_list, exclude=None, scaling_factor=2000):
    """
    Compute QI after dropping one nutrient from nutrient_list
    """
    if exclude is not None:
        new_list = [nutr for nutr in nutrient_list if nutr != exclude]
    else:
        new_list = nutrient_list
    

    return compute_index(row, new_list, scaling_factor=scaling_factor)



def compare_qi_excluding_nutrient(df, nutrient_to_exclude, qualifying_nutrients, new_col_name=None,scaling_factor=2000):
    """
    For each food item, compare QI w/ & w/o one qualifying nutrient
    Plot a bar chart, and return a DataFramw w/ differences
    """

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
    """
    Compute QI after dropping multiple nutrients in exclude_list
    """
    new_list = nutrient_list.copy()  

    for nutr in exclude_list:
        if nutr in new_list:
            new_list.remove(nutr)

    return compute_index(row, new_list, scaling_factor=scaling_factor)


def plot_nutrient_contributions_with_qi(row, nutrient_cols, exclude_list=None, scaling_factor=2000):
    """
    Plot bar plot of each nutrient's scaled ratio for a single food item, annotated w/ its contribution to QI
    """
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
    """
    Compute NB after dropping one nutrient
    """
    if exclude is not None:
        new_list = [nutr for nutr in nutrient_list if nutr != exclude]
    else:
        new_list = nutrient_list
    return compute_nb(row, new_list, scaling_factor=scaling_factor)
    

def compare_nb_excluding_nutrient(df, nutrient_to_exclude, qualifying_nutrients, new_col_name=None, scaling_factor=2000):
    """
    Similar to compare_qi_excluding_nutrient but for NB
    """
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
    """
    Compute NB after dropping multiple nutrients
    """
    new_list = nutrient_list.copy()  
    for nutr in exclude_list:
        if nutr in new_list:
            new_list.remove(nutr)
    return compute_nb(row, new_list, scaling_factor=scaling_factor)

def compare_di_excluding_nutrient(df, nutrient_to_exclude, disqualifying_nutrients, new_col_name=None, scaling_factor=2000):
    """
    Similar to compare_qi_excluding_nutrient but for DI
    """
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


def remove_outliers(df, column, factor=5):
    """
    Drop rows that are above a certain treshold (= median + factor * IQR)
    Return a DataFrame
    """
 
    med = df[column].median()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    threshold = med + factor * IQR
    df_clean = df[df[column] <= threshold]
    return df_clean, threshold


def threshold_median_iqr(df, column, factor=5):
    """
    Compute treshold
    """

    med = df[column].median()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    threshold = med + factor * IQR
    return threshold

def weighted_mean(values, weights):
    """
    Compute weignted average
    """
    if weights.sum() != 0:
        return (values * weights).sum() / weights.sum()
    else:
        return np.nan



def classify_meal_time(dt):
    """
    Assign meal label based on hour of day
    """

    hour = dt.hour
    if 7 <= hour < 10:
        return "breakfast"
    elif 11 <= hour < 14:
        return "lunch"
    elif 15 <= hour < 17:
        return "snack"
    elif 18 <= hour < 22:
        return "dinner"
    else :
        return None
    
def plot_meal(subject_meals, subject_id, meal_name, time_column='meal_time', save_path=None):
    """
    Plots composite meal scores over time for a given subject
    """

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(subject_meals[time_column], subject_meals['QI'], marker='o', linestyle='-', color='blue', label='QI')
    ax1.plot(subject_meals[time_column], subject_meals['DI'], marker='x', linestyle='-', color='orange', label='DI')
    ax1.set_xlabel('Meal Date')
    ax1.set_ylabel('QI and DI', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.bar(subject_meals[time_column], subject_meals['NB'], width=0.3, color='lightgrey', alpha=0.5, label='NB')
    ax2.set_ylabel('NB (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(f'Composite "{meal_name.capitalize()}" Scores Over Time for Subject: {subject_id}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()

def plot_meal_composite(df, subject_id, target_date, meal_name, save_path=None):

    df_meal = df[(df['subject_key'] == subject_id) & 
                 (df['date'] == target_date) &
                 (df['meal'] == meal_name)].copy()
    
    if df_meal.empty:
        print(f"No data for subject {subject_id} on {target_date} for {meal_name}.")
        return
    
    df_meal.sort_values('eaten_at', inplace=True)
    
    total_energy = df_meal['energy_kcal_eaten'].sum()
    df_meal['energy_pct'] = df_meal['energy_kcal_eaten'] / total_energy * 100
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df_meal['QI'], df_meal['DI'], s=400, color='black', zorder=4)
    
    for i, row in df_meal.iterrows():
        x = row['QI']
        y = row['DI']
        nb_val = row['NB']
        name = row['combined_name']
        plt.text(x, y, f"{nb_val:.1f}", ha='center', va='center', color='white', fontsize=9, zorder=5)
        plt.text(x, y + 0.08, name, ha='center', va='bottom', color='black', fontsize=8, rotation=0, zorder=5)
    
    # Compute the composite (weighted) meal scores:
    comp_qi = weighted_mean(df_meal['QI'], df_meal['energy_kcal_eaten'])
    comp_di = weighted_mean(df_meal['DI'], df_meal['energy_kcal_eaten'])
    comp_nb = weighted_mean(df_meal['NB'], df_meal['energy_kcal_eaten'])
    
    # Compute angles for each point relative to the composite point
    angles = np.arctan2(df_meal['DI'] - comp_di, df_meal['QI'] - comp_qi)
    df_meal_sorted = df_meal.copy()
    df_meal_sorted['angle'] = angles
    df_meal_sorted.sort_values('angle', inplace=True)
    
    loop_qi = df_meal_sorted['QI'].tolist()
    loop_di = df_meal_sorted['DI'].tolist()
    loop_qi.append(loop_qi[0])
    loop_di.append(loop_di[0])
    
    plt.plot(loop_qi, loop_di, color='grey', linestyle='--', alpha=0.7, zorder=3)
    
    plt.scatter(comp_qi, comp_di, s=600, color='red', zorder=6)
    #plt.text(comp_qi, comp_di, f"Combined Meal\n{weighted_mean(df_meal['NB'], df_meal['energy_kcal_eaten']):.1f}", ha='center', va='center', color='black', fontsize=10, zorder=7)
    plt.text(comp_qi, comp_di, f"Combined Meal\nNB: {comp_nb:.1f}", ha='center', va='center', color='black', fontsize=10, zorder=7)

    plt.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.xlabel('Qualifying Index (QI)')
    plt.ylabel('Disqualifying Index (DI)')
    plt.title(f"Composite {meal_name.capitalize()} for Subject {subject_id} on {target_date}")
    plt.xlim(0, 8)
    plt.ylim(0, 4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_day_meals(subject_day_df, subject_id, target_date, save_path=None):
    """
    Plots composite meal scores for a given subject on a specific day.
    """
    meal_order = ['breakfast', 'lunch', 'snack', 'dinner']

    df_day = subject_day_df[subject_day_df['meal'].isin(meal_order)].copy()
    
    df_day['meal'] = pd.Categorical(df_day['meal'], categories=meal_order, ordered=True)
    df_day.sort_values('meal', inplace=True)

    x = np.arange(len(df_day))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(x, df_day['QI'], marker='o', linestyle='-', color='blue', label='QI')
    ax1.plot(x, df_day['DI'], marker='x', linestyle='-', color='orange', label='DI')
    ax1.set_xlabel("Meal Type")
    ax1.set_ylabel("Composite QI and DI", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_day['meal'], rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.bar(x, df_day['NB'], width=0.3, color='lightgrey', alpha=0.5, label='NB')
    ax2.set_ylabel("Composite NB (%)", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
 
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    

    plt.title(f"Composite Meal Scores for Subject {subject_id} on {target_date}")
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()
    
    return df_day

def plot_daily_qi_di_boxplots(composite_df, meal_type=None, save_path=None):
    """
    Boxplots of QI & DI weekday name
    """

    # Make a copy and filter by meal type
    df = composite_df.copy()
    if meal_type is not None:
        df = df[df['meal'] == meal_type]
    
    # Convert date to datetime and extract day-of-week
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
    
    df_long = pd.melt(df, 
                      id_vars=['day_of_week'],
                      value_vars=['QI', 'DI'],
                      var_name='Score_Type',
                      value_name='Score')

    plt.figure(figsize=(10, 6))
    

    custom_palette = {'QI': 'skyblue', 'DI': 'deeppink'}
    sns.boxplot(x='day_of_week', y='Score', hue='Score_Type', data=df_long, palette=custom_palette)
    
    plt.xlabel("Day of the Week")
    plt.ylabel("QI, DI")
    if meal_type is not None:
        plt.title(f"Composite QI and DI Distributions by Day for {meal_type.capitalize()}")
    else:
        plt.title("Composite QI and DI Distributions by Day (All Meals)")
    
    plt.legend(title="Score Type", loc="upper left")
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    
    return

def add_type_of_day(df, date_col='date'):
    """
    Adds two columns :
        - day_of_weeks
        - type_of_day ('weekday' vs 'weekend')
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.day_name()

    # map Mon–Fri → 'weekday', Sat/Sun → 'weekend'
    df['type_of_day'] = df['day_of_week'].isin(['Saturday','Sunday']).map({True:'weekend', False:'weekday'})
    return df



def test_weekday_vs_weekend(composite_df, min_energy=100):
    """
    Runs a Mann-Whitney U test on QI and DI between weekday and weekend (non-parametric)
    Return a DataFram of U statistic and p-value
    """

    df = composite_df.copy()
    results = []
    
    for meal in df['meal'].unique():
        sub = df[df['meal'] == meal]
        sub = sub[sub['total_energy'] >= min_energy]
        
    
        for metric in ['QI', 'DI']:
            wkday = sub.loc[sub['type_of_day'] == 'weekday', metric].dropna()
            wkend = sub.loc[sub['type_of_day'] == 'weekend', metric].dropna()
            if len(wkday) < 5 or len(wkend) < 5:
                continue
            
            U, p = mannwhitneyu(wkday, wkend, alternative='two-sided')
            results.append({
                'meal': meal,
                'metric': metric,
                'weekday_n': len(wkday),
                'weekend_n': len(wkend),
                'U_stat': U,
                'p_value': p
            })
    
    return pd.DataFrame(results)

def star(p):
    """
    Convert p-value into signifiance stars
    """
    if p <= 1e-4: return "****"
    if p <= 1e-3: return "***"
    if p <= 1e-2: return "**"
    if p <= 5e-2: return "*"
    return "ns"


def plot_qi_di_by_meal(grouped, agg='mean', save_path=None):
    """
    Plot daily QI & DI (solid/dashed) by meal type

    agg:    aggregation key ('mean' or 'median')
    """

    df = grouped.copy()
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], days, ordered=True)

    fig, ax = plt.subplots(figsize=(10,6))

    sns.lineplot(
        data=df,
        x='day_of_week',
        y=f'QI_{agg}',
        hue='meal',
        marker='o',
        ax=ax
    )
   
    sns.lineplot(
        data=df,
        x='day_of_week',
        y=f'DI_{agg}',
        hue='meal',
        marker='X',
        linestyle='--',
        ax=ax,
        legend=False
    )

    ax.set_xlabel('')
    ax.set_ylabel(f'QI & DI ({agg})')
    ax.tick_params(axis='x', rotation=45)
    ax.set_title(f'Daily QI & DI ({agg.capitalize()}) by Meal')

    # meal' legend
    meal_legend = ax.legend(title='Meal', loc='upper left')

    style_handles = [
        Line2D([0],[0], color='black', lw=2, linestyle='-'),
        Line2D([0],[0], color='black', lw=2, linestyle='--'),
    ]
    style_labels = ['QI', 'DI']

    ax.add_artist(meal_legend)
    ax.legend(
        handles=style_handles,
        labels=style_labels,
        title='Metric',
        loc='upper right'
    )

    if save_path is not None:
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()

    return fig, ax


def boxplot_qi_di_by_meal_for_day(df,
                               day_of_week,
                               meal_order=None,
                               save_path=None):
    """
    For a given weekday boxplots of QI & DI for each of the four meal types on that day.

    """
    df2 = df[df['day_of_week']==day_of_week].copy()

    # determine meal order
    if meal_order is None:
        meal_order = sorted(df2['meal'].unique())
    df2['meal'] = pd.Categorical(df2['meal'],
                                 categories=meal_order,
                                 ordered=True)


    long = df2[['meal','QI','DI']].melt(
        id_vars='meal',
        value_vars=['QI','DI'],
        var_name='Metric',
        value_name='Score'
    )

    plt.figure(figsize=(10,5))
    sns.boxplot(data=long,
                x='meal',
                y='Score',
                hue='Metric',
                palette={'QI':'skyblue','DI':'deeppink'})
    plt.title(f"{day_of_week} – QI & DI by Meal")
    plt.xlabel("") 
    plt.ylabel("Score")
    plt.legend(title="")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out = save_path if save_path.endswith(".png") else save_path + ".png"
        plt.savefig(out, dpi=150)
        print(f"Saved to {out}")

    plt.show()



def aggregate_meal_nutrients_fast(df, nutrient_cols):
    """
    Vectorized aggregation: for each (subject_key, date, meal) :
    - sum_energy_for_nutrients = total energy
    - for each nutrient: energy‐weighted mean of nutrient_ratio_scaled
    """

    df2 = df.copy()
    group_cols = ['subject_key','date','meal']
    energy = 'energy_kcal_eaten'

  
    weighted_cols = []
    for n in nutrient_cols:
        col = n + '_ratio_scaled'
        wcol = 'w_' + col
        df2[wcol] = df2[col] * df2[energy]
        weighted_cols.append(wcol)

    # group‐energy and weighted columns
    agg_dict = {energy: 'sum'}
    agg_dict.update({w: 'sum' for w in weighted_cols})

    grouped = df2.groupby(group_cols).agg(agg_dict)
    grouped = grouped.rename(columns={energy: 'sum_energy_for_nutrients'})

    # compute weighted mean
    for n in nutrient_cols:
        col = n + '_ratio_scaled'
        wcol = 'w_' + col
        grouped[col] = grouped[wcol] / grouped['sum_energy_for_nutrients']

    # drop intermediate weighted sums
    grouped = grouped.drop(columns=weighted_cols).reset_index()

    return grouped

def compute_qi_excluding_scaled(row, nutrient_list, exclude=None):
    """
    Compute QI purely as the mean of all row[nutr+'_ratio_scaled'] except the one named by `exclude`.
    """

    if exclude is not None:
        nuts = [n for n in nutrient_list if n != exclude]
    else:
        nuts = nutrient_list

    cols = [f"{n}_ratio_scaled" for n in nuts]
    return row[cols].mean()

def compute_di_excluding_scaled(row, nutrient_list, exclude=None):
    """
    same function as compute_qi_excluding_scaled but for di
    """

    nuts = [n for n in nutrient_list if n!=exclude]
    cols = [f"{n}_ratio_scaled" for n in nuts]
    return row[cols].mean()



def rank_nutrient_impact(df, score_col, nutrient_list, exclude_fn):
    """
    For each nutrient in nutrient_list, compute the mean drop in `score_col` when nutr is excluded
    """

    impacts = {}
    for nut in nutrient_list:
        excl_scores = df.apply(lambda r: exclude_fn(r, nutrient_list, exclude=nut), axis=1)
        impacts[nut] = (df[score_col] - excl_scores).mean()
    return pd.Series(impacts).sort_values(ascending=False)


def rank_nutrient_impact_fast(df, score_col, nutrient_list):
    """
    Vectorized: for each nutrient in nutrient_list, compute the average drop in score_col when that nutrient is excluded from the simple mean.
    """

    p = len(nutrient_list)
   
    ratio_cols = [f"{n}_ratio_scaled" for n in nutrient_list]
    mean_ratios = df[ratio_cols].mean()
    mean_score  = df[score_col].mean()
    impacts = (mean_ratios - mean_score) / (p - 1)
    impacts.index = nutrient_list
    
    return impacts.sort_values(ascending=False)



def plot_impact_correlation(
    df_meal_nutrient,
    score_col: str,
    nutrient_list: list[str],
    exclude_fn,                
    figsize=(8,8),
    cmap="vlag",
    save_path: str|None = None
):
    """
    Clustered heatmap of correlations among score vectors for each nutrient
    """
    # assemble scaled-ratio matrix and original score
    cols = [f"{n}_ratio_scaled" for n in nutrient_list]
    R = df_meal_nutrient[cols].to_numpy()        
    orig = df_meal_nutrient[score_col].to_numpy()  
    m = R.shape[1]

    # leave-one-out means & score
    row_sums = R.sum(axis=1, keepdims=True)       
    LOO = (row_sums - R) / (m - 1)                
    delta = orig.reshape(-1,1) - LOO              

    drop_df = pd.DataFrame(delta, columns=nutrient_list, index=df_meal_nutrient.index)
    corr = drop_df.corr()

    g = sns.clustermap(
        corr,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        figsize=figsize,
        cbar_kws={"label": f"corr(Δ{score_col},Δ{score_col})"}
    )
    g.ax_heatmap.set_title(f"Correlation of Δ{score_col} impact vectors")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return corr


def nutrient_correlations(df, nutrient_list, score_col):
    """
    Computes the absolute Pearson r between each nutrients scaled ratio and the index
    """

    ratio_cols = [f"{n}_ratio_scaled" for n in nutrient_list]
    sub = df[df['energy_kcal_eaten'] > 0]  
    corr = sub[ ratio_cols + [score_col] ].corr()[score_col].drop(score_col)
    return corr.abs().sort_values(ascending=False)



def qi_drop_mean(df, nut):
    """
    Recompute each meal's QI with exactly one nutrient removed, then measure the average drop in QI 
    """

    orig = df["QI"]
    cols = [c for c in df.columns if c.endswith("_ratio_scaled") and not c.startswith(f"{nut}_ratio_scaled")]
    new_qi = df[cols].mean(axis=1)
    return (orig - new_qi).mean()

def di_drop_mean(df, nut):
    """
    Recompute each meal's QI with exactly one nutrient removed, then measure the average drop in QI 
    """

    orig = df["DI"]
    cols = [c for c in df.columns if c.endswith("_ratio_scaled") and not c.startswith(f"{nut}_ratio_scaled")]
    new_di = df[cols].mean(axis=1)
    return (orig - new_di).mean()


def significance_test(data, group_col, value_col, alpha=0.05, non_parametric=False):
    groups = data[group_col].unique()
    if len(groups) == 2:
        group1 = data[data[group_col] == groups[0]][value_col]
        group2 = data[data[group_col] == groups[1]][value_col]
        if non_parametric:
            stat, p_val = mannwhitneyu(group1, group2)
            test_name = "Mann-Whitney U"
        else:
            stat, p_val = ttest_ind(group1, group2)
            test_name = "t-test"
        return p_val < alpha, p_val, test_name, stat
    elif len(groups) > 2:
        group_values = [data[data[group_col] == grp][value_col] for grp in groups]
        if non_parametric:
            stat, p_val = kruskal(*group_values)
            test_name = "Kruskal-Wallis"
        else:
            stat, p_val = f_oneway(*group_values)
            test_name = "ANOVA"
        return p_val < alpha, p_val, test_name, stat
    return False, 1.0, "No test", 0
def create_quartiles(df, column_name, ncut=4):
    """Add quartile categories (1-4) for a specified column in a DataFrame."""
    df_copy = df.copy()
    df_copy[column_name + "_quartile"] = pd.qcut(
        df_copy[column_name], q=ncut, labels=[f"Q{i}" for i in range(1, ncut + 1)]
    )
    return df_copy
def add_stat_annotation(ax, data, x, y, pairs, test="Mann-Whitney", text_format="star"):
    """Add statistical annotation bars to the plot."""
    annotator = Annotator(ax, pairs, data=data, x=x, y=y)
    annotator.configure(
        test=test,
        text_format=text_format,
        line_width=0.5,  # Thinner lines
        line_height=0.02,
    )  # Slightly shorter bars
    annotator.apply_and_annotate()
def test_quartile_differences(
    df,
    column_name,
    value_col,
    extremes=True,
    alpha=0.05,
    non_parametric=True,
    ax=None,
    palette="Spectral",
    make_quartile=True,
    verbose=False,
):
    """
    Test differences between quartiles of a specified column and create a boxplot with significance bars.
    """
    df = df.copy()
    if make_quartile:
        df = create_quartiles(df, column_name)
        quartile_col = column_name + "_quartile"
    else:
        quartile_col = column_name
    if extremes:
        df = df[df[quartile_col].isin(["Q1", "Q4"])]
        if isinstance(df[quartile_col].dtype, pd.CategoricalDtype):
            df[quartile_col] = df[quartile_col].cat.remove_unused_categories()
    # Create plot if ax not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    # Create boxplot
    sns.boxplot(
        data=df,
        x=quartile_col,
        y=value_col,
        ax=ax,
        linewidth=0.5,
        fliersize=1,
        width=0.5,
        palette=palette,
    )
    # Get unique groups and create pairs for comparison
    groups = sorted(df[quartile_col].dropna().astype(str).unique())
    if extremes:
        # If only comparing extremes, just compare Q1 and Q4
        pairs = [("Q1", "Q4")]
    else:
        # Create all possible pairs of quartiles
        pairs = list(combinations(groups, 2))
    # Determine statistical test
    test = "Mann-Whitney" if non_parametric else "t-test"
    # Add statistical annotation
    add_stat_annotation(ax, df, x=quartile_col, y=value_col, pairs=pairs, test=test)
    # Run the original significance test for console output
    is_sig, p_val, test_name, stat = significance_test(
        df, quartile_col, value_col, alpha=alpha, non_parametric=non_parametric
    )
    if verbose:
        if is_sig:
            print(
                f"{column_name} -> {value_col}\tSignificant differences exist.",
                f"{test_name} statistic: {stat:.3f} p-value: {p_val:.3f}",
            )
        else:
            print(f"{column_name} -> {value_col}\tNo significant differences.")
    # Add test statistics to plot
    ylims = ax.get_ylim()
    yrange = ylims[1] - ylims[0]
    text_y = ylims[1] + yrange * 0.05  # Position text above the plot
    # Format test information with abbreviated names
    test_abbrev = {
        "Kruskal-Wallis": "KW",
        "Mann-Whitney U": "MW",
        "ANOVA": "ANOVA",
        "t-test": "t-test",
    }
    abbreviated_name = test_abbrev.get(test_name, test_name)
    if test_name == "Kruskal-Wallis":
        test_info = f"${abbreviated_name}:$ $H={stat:.2f}$, $p={p_val:.3f}$"
    elif test_name == "Mann-Whitney U":
        test_info = f"${abbreviated_name}:$ $U={stat:.2f}$, $p={p_val:.3f}$"
    else:
        test_info = f"${abbreviated_name}:$ $t={stat:.2f}$, $p={p_val:.3f}$"
    # Add text with test information
    ax.text(
        0,
        text_y,
        test_info,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=8,
    )
    # Adjust plot limits 
    # to accommodate the text
    ax.set_ylim(ylims[0], ylims[1] + yrange * 0.15)
    return ax


def plot_qi_nb_scatter(subject_id, df, start_date=None, end_date=None):
    

    subject_df = df[df['subject_key'] == subject_id].copy()
    
    # Look at specific time frame
    if start_date is not None:
        subject_df = subject_df[subject_df['date'] >= start_date]
    if end_date is not None:
        subject_df = subject_df[subject_df['date'] <= end_date]

  
    daily_agg = (subject_df.groupby('date')[['QI', 'NB']].median().reset_index())
    

    plt.figure(figsize=(10, 8))
    
    # Individual meal components
    plt.scatter(subject_df['QI'], subject_df['NB'], color='black', alpha=0.7, label='Individual Meal Components')
    
    # Daily aggregated values 
    plt.scatter(daily_agg['QI'], daily_agg['NB'], color='red', s=80, label='Daily Combined Meals')
    
    
    plt.axvline(x=1, color='black', linestyle='--', alpha=1)
   
    plt.xlabel('Qualifying Index (QI)')
    plt.ylabel('% Nutrient Balance (NB)')
    plt.xlim(0, 14)
    plt.title(f'QI vs NB for Subject : {subject_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(daily_agg)


def nonparametric_group_comparison(df, value_col, group_col='food_group', alpha=0.05, p_adjust_method='bonferroni'):
    """
    Performs a Kruskal-Wallis test across groups and pairwise Mann-Whitney U tests with p-value adjustment
    
    Parameters:
    - df: pandas DataFrame
    - value_col: column name for the metric
    - group_col: 
    - alpha: significance level 
    - p_adjust_method: method for p-value adjustment 
    
    Returns:
    - H: Kruskal-Wallis H statistic
    - p_kruskal: Kruskal-Wallis p-value
    - posthoc_df: DataFrame with pairwise comparisons and adjusted p-values

    """
    # Kruskal–Wallis test
    groups = [grp[value_col].values for _, grp in df.groupby(group_col)]
    H, p_kruskal = kruskal(*groups)
    print(f"Kruskal-Wallis H for {value_col} = {H:.2f}, p = {p_kruskal:.3g}")
    
    # Pairwise Mann–Whitney U tests
    group_dict = {name: grp[value_col].values for name, grp in df.groupby(group_col)}
    pairs = list(itertools.combinations(group_dict.keys(), 2))
    results = []
    for g1, g2 in pairs:
        u_stat, p_unc = mannwhitneyu(group_dict[g1], group_dict[g2], alternative='two-sided')
        results.append({
            'group1': g1, 'group2': g2,
            'U_stat': u_stat, 'p_uncorrected': p_unc
        })
    
    # P-value adjustment
    p_unc_list = [r['p_uncorrected'] for r in results]
    rej, p_adj, _, _ = multipletests(p_unc_list, alpha=alpha, method=p_adjust_method)
    for r, p_a, sig in zip(results, p_adj, rej):
        r['p_adjusted'] = p_a
        r['significant'] = sig
    
    posthoc_df = pd.DataFrame(results).sort_values('p_adjusted')
    print(f"\nPairwise Mann-Whitney U with {p_adjust_method} adjustment:")
    display(posthoc_df[['group1','group2','U_stat','p_uncorrected','p_adjusted','significant']])
    
    return H, p_kruskal, posthoc_df

