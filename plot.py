# %% [markdown]
# ## Import and Configuration

# %%
import os
import re
import csv
import json
import math
import openai
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from datetime import datetime
from collections import defaultdict, Counter
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap

# %%
def extract_brackets(text, brackets='[]'):
    assert len(brackets) == 2
    pattern = re.escape(brackets[0]) + r'(.*?)' + re.escape(brackets[1])
    matches = re.findall(pattern, text)
    return matches

def extract_amout(
    message, 
    prefix='',
    print_except=True,
    type=float,
    brackets='[]'
):
    try:
        matches = extract_brackets(message, brackets=brackets)
        matches = [s[len(prefix):] \
            if s.startswith(prefix) \
            else s for s in matches]
        invalid = False
        if len(matches) == 0:
            invalid = True
        for i in range(len(matches)):
            if matches[i] != matches[0]:
                invalid = True
        if invalid:
            raise ValueError('Invalid answer: %s' % message)
        return type(matches[0])
    except Exception as e: 
        if print_except: print(e)
        return None

def extract_choices(recrods):
    choices = [extract_amout(
        messages[-1]['content'], 
        prefix='$', 
        print_except=True,
        type=float) for messages in records['messages']
    ]
    choices = [x for x in choices if x is not None]
    # print(choices)
    return choices

# %%
def choices_to_df(choices, hue):
    df = pd.DataFrame(choices, columns=['choices'])
    df['hue'] = hue
    df['hue'] = df['hue'].astype(str)
    return df

def plot(
        df, 
        title='',
        x='choices',
        hue='hue',
        binrange=None, 
        binwidth=None,
        stat='count',
        multiple='dodge'
    ):
    if binrange is None:
        binrange = (df[x].min(), df[x].max())
    df = df.dropna(axis=0, subset=[x]).reset_index()
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(
        data=df, 
        x=x,
        hue=hue, 
        kde=True,
        binrange=binrange, 
        binwidth=binwidth,
        stat=stat,
        multiple=multiple,
        shrink=.8,
    )
    ax.set_title(title)
    return ax

# %%
def plot_facet(
    df_list,
    x='choices',
    hue='hue',
    palette=None,
    binrange=None,
    bins=10,
    # binwidth=10,
    stat='count',
    x_label='',
    sharex=True,
    sharey=False,
    subplot=sns.histplot,
    xticks_locs=None,
    # kde=False,
    **kwargs
):
    data = pd.concat(df_list)
    if binrange is None:
        binrange = (data[x].min(), data[x].max())
    g = sns.FacetGrid(
        data, row=hue, hue=hue, 
        palette=palette,
        aspect=2, height=2, 
        sharex=sharex, sharey=sharey,
        despine=True,
    )
    g.map_dataframe(
        subplot, 
        x=x, 
        # kde=kde, 
        binrange=binrange, 
        bins=bins,
        stat=stat,
        **kwargs
    )
    # g.add_legend(title='hue')
    g.set_axis_labels(x_label, stat.title())
    g.set_titles(row_template="{row_name}")
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos: '{:.2f}'.format(y))
        )
    
    binwidth = (binrange[1] - binrange[0]) / bins
    if xticks_locs is None:
        locs = np.linspace(binrange[0], binrange[1], bins//2+1)
        locs = [loc + binwidth for loc in locs]
    else: 
        locs = xticks_locs
    labels = [str(int(loc)) for loc in locs]
    locs = [loc + 0.5*binwidth for loc in locs]
    plt.xticks(locs, labels)
    
    g.set(xlim=binrange)
    return g

# %%
sns.set(rc={'figure.figsize':(5,4)})
sns.set_style("ticks")

default_palette = sns.color_palette(None)
blue = default_palette[0]
orange = default_palette[1]
green = default_palette[2]
red = default_palette[3]
purple = default_palette[4]

# %%
def plot_occupations(
    df_baseline,
    choices_all,
    binrange=(0,100),
    binwidth=5,
    x_label='$ to give',
    stat='density',
    model='ChatGPT-4',
):
    print('baseline: ', len(df_baseline))
    df_list = []
    for occupation in choices_all:
        choices = choices_all[occupation]
        print(occupation, ':', len(choices))
        df = choices_to_df(choices, hue='{} ({})'.format(model, occupation))
        df_list.append(df)
    g = plot_facet(
        df_list=[df_baseline]+df_list,
        binrange=binrange,
        binwidth=binwidth,
        x_label=x_label,
        stat=stat,
    )
    return g

# %% [markdown]
# ## Fig. 2: Turing Test
# 
# Results may vary due to randomness in the simulation.

# %%
# def simulate_Turing(samples_0, samples_1, n_bin=10, lim_a=0, lim_b=100, n_draw=100000):
#     hist_0 = np.histogram(samples_0, bins=n_bin, range=(lim_a, lim_b))[0] / len(samples_0)
#     n_wins = 0
#     n_ties = 0
#     for _ in tqdm(range(n_draw)):
#         try:
#             sample_0 = np.random.choice(samples_0)
#             sample_1 = np.random.choice(samples_1)
#             idx_0 = min(math.floor((sample_0 - lim_a) / (lim_b - lim_a) * n_bin), n_bin-1)
#             idx_1 = min(math.floor((sample_1 - lim_a) / (lim_b - lim_a) * n_bin), n_bin-1)
#             if hist_0[idx_1] > hist_0[idx_0]:
#                 n_wins += 1
#             elif hist_0[idx_1] == hist_0[idx_0]:
#                 n_ties += 1
#         except:
#             continue
#     return n_wins / n_draw, n_ties / n_draw, (n_draw - n_wins - n_ties) / n_draw

# %%
# print('human', simulate_Turing(df_dictator_human['choices'], df_dictator_human['choices']))
# print('gpt4', simulate_Turing(df_dictator_human['choices'], df_dictator_gpt4['choices']))
# print('turbo', simulate_Turing(df_dictator_human['choices'], df_dictator_turbo['choices']))

# %%
# print('human', simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_human['choices']))
# print('gpt4', simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_gpt4['choices']))
# print('turbo', simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_turbo['choices']))

# %%
# print('human', simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_human['choices']))
# print('gpt4', simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_gpt4['choices']))
# print('turbo', simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_turbo['choices']))

# %%
# print('human', simulate_Turing(df_trust_1_human['choices'], df_trust_1_human['choices']))
# print('gpt4', simulate_Turing(df_trust_1_human['choices'], df_trust_1_gpt4['choices']))
# print('turbo', simulate_Turing(df_trust_1_human['choices'], df_trust_1_turbo['choices']))

# %%
# print('human', simulate_Turing(df_trust_3_human['choices'], df_trust_3_human['choices'], lim_b=150))
# print('gpt4', simulate_Turing(df_trust_3_human['choices'], df_trust_3_gpt4['choices'], lim_b=150))
# print('turbo', simulate_Turing(df_trust_3_human['choices'], df_trust_3_turbo['choices'], lim_b=150))

# %%
# # Convert to numeric, coercing errors (invalid parsing will be set as NaN)
# samples_0 = pd.to_numeric(df_PG_human['choices'], errors='coerce')
# samples_1 = pd.to_numeric(df_PG_gpt4['choices'], errors='coerce')

# %%
# df_PG_turbo['choices'].dtype

# %%
# print(df_PG_turbo['choices'].head())

# %%


# nan_count = df_PG_turbo['choices'].isna().sum()
# print(f"Number of NaN values: {nan_count}")

# df_PG_turbo['choices'].fillna(0, inplace=True)

# %%
# df_PG_turbo['choices'] = df_PG_turbo['choices'].dropna()

# %%
# print('human', simulate_Turing(df_PG_human['choices'], df_PG_human['choices'], lim_b=20))
# print('gpt4', simulate_Turing(df_PG_human['choices'], df_PG_gpt4['choices'], lim_b=20))
# print('gpt4', simulate_Turing(df_PG_turbo['choices'], df_PG_turbo['choices'], lim_b=20))

# %%
# print('human', simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_human['']))
# print('gpt4', simulate_Turing(prefix_to_choices_human[''],prefix_to_choices_model['ChatGPT-4']['']))
# print('turbo', simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_model['ChatGPT-3']['']))

# %%
# print('human', r_coo_human * r_def_human, r_coo_human * r_coo_human + r_def_human * r_def_human, r_def_human * r_coo_human)
# print('gpt4', r_coo_human * r_def_gpt4, r_coo_human * r_coo_gpt4 + r_def_human * r_def_gpt4, r_def_human * r_coo_gpt4)
# print('turbo', r_coo_human * r_def_turbo, r_coo_human * r_coo_turbo + r_def_human * r_def_turbo, r_def_human * r_coo_turbo)

# n_coo_human = 36269
# n_def_human = 44114
# r_coo_human = n_coo_human / (n_coo_human + n_def_human)
# r_def_human = n_def_human / (n_coo_human + n_def_human)
# print(r_coo_human, r_def_human)

# n_coo_gpt4 = 29 + 0 + 0 + 26
# n_def_gpt4 = 0 + 1 + 1 + 3
# n_coo_turbo = 21 + 3 + 7 + 15
# n_def_turbo = 3 + 3 + 4 + 4
# r_coo_gpt4 = n_coo_gpt4 / (n_coo_gpt4 + n_def_gpt4)
# r_def_gpt4 = n_def_gpt4 / (n_coo_gpt4 + n_def_gpt4)
# r_coo_turbo = n_coo_turbo / (n_coo_turbo + n_def_turbo)
# r_def_turbo = n_def_turbo / (n_coo_turbo + n_def_turbo)
# print(r_coo_gpt4, r_def_gpt4)
# print(r_coo_turbo, r_def_turbo)

# %%


import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import pandas as pd

# player_average('human', df_dictator_human['choices'])
# player_average('gpt4', df_dictator_gpt4['choices'])
# player_average('turbo', df_dictator_turbo['choices'])

# player_average('human', [1]*n_coo_human + [0]*n_def_human)
# player_average('gpt4', [1]*n_coo_gpt4 + [0]*n_def_gpt4)
# player_average('turbo', [1]*n_coo_turbo + [0]*n_def_turbo)


# Average_human_result = simulate_Turing(df_average_human['choices'], df_average_human['choices'])
# Average_gpt4_result = simulate_Turing(df_average_human['choices'], df_average_gpt4['choices'])
# Average_turbo_result = simulate_Turing(df_average_human['choices'], df_average_turbo['choices'])




# Dictator_human_result = simulate_Turing(df_dictator_human['choices'], df_dictator_human['choices'])
# Dictator_gpt4_result = simulate_Turing(df_dictator_human['choices'], df_dictator_gpt4['choices'])
# Dictator_turbo_result = simulate_Turing(df_dictator_human['choices'], df_dictator_turbo['choices'])

# Ultimatum_1_human_result = simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_human['choices'])
# Ultimatum_1_gpt4_result = simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_gpt4['choices'])
# Ultimatum_1_turbo_result = simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_turbo['choices'])

# Ultimatum_2_human_result = simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_human['choices'])
# Ultimatum_2_gpt4_result = simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_gpt4['choices'])
# Ultimatum_2_turbo_result = simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_turbo['choices'])

# Trust_1_human_result = simulate_Turing(df_trust_1_human['choices'], df_trust_1_human['choices'])
# Trust_1_gpt4_result = simulate_Turing(df_trust_1_human['choices'], df_trust_1_gpt4['choices'])
# Trust_1_turbo_result = simulate_Turing(df_trust_1_human['choices'], df_trust_1_turbo['choices'])

# Trust_3_human_result = simulate_Turing(df_trust_3_human['choices'], df_trust_3_human['choices'], lim_b=150)
# Trust_3_gpt4_result = simulate_Turing(df_trust_3_human['choices'], df_trust_3_gpt4['choices'], lim_b=150)
# Trust_3_turbo_result = simulate_Turing(df_trust_3_human['choices'], df_trust_3_turbo['choices'], lim_b=150)

# Public_Goods_human_result = simulate_Turing(df_PG_human['choices'], df_PG_human['choices'], lim_b=20)
# Public_Goods_gpt4_result = simulate_Turing(df_PG_human['choices'], df_PG_gpt4['choices'], lim_b=20)
# Public_Goods_turbo_result = simulate_Turing(df_PG_turbo['choices'], df_PG_turbo['choices'], lim_b=20)

# Bomb_Risk_human_result = simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_human[''])
# Bomb_Risk_gpt4_result = simulate_Turing(prefix_to_choices_human[''],prefix_to_choices_model['ChatGPT-4'][''])
# Bomb_Risk_turbo_result = simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_model['ChatGPT-3'][''])

# Prisoner_Dilemma_human_result = r_coo_human * r_def_human, r_coo_human * r_coo_human + r_def_human * r_def_human, r_def_human * r_coo_human
# Prisoner_Dilemma_gpt4_result = r_coo_human * r_def_gpt4, r_coo_human * r_coo_gpt4 + r_def_human * r_def_gpt4, r_def_human * r_coo_gpt4
# Prisoner_Dilemma_turbo_result = r_coo_human * r_def_turbo, r_coo_human * r_coo_turbo + r_def_human * r_def_turbo, r_def_human * r_coo_turbo


# data = {
#     # 'Average': {
#     #     'Human': Average_human_result,
#     #     'GPT-4': Average_gpt4_result,
#     #     'Turbo': Average_turbo_result,
#     # },
#     'Dictator': {
#         'Human': Dictator_human_result,
#         'GPT-4': Dictator_gpt4_result,
#         'Turbo': Dictator_turbo_result,
#     },
#     'Ultimatum 1': {
#         'Human': Ultimatum_1_human_result,
#         'GPT-4': Ultimatum_1_gpt4_result,
#         'Turbo': Ultimatum_1_turbo_result,
#     },
#     'Ultimatum 2': {
#         'Human': Ultimatum_2_human_result,
#         'GPT-4': Ultimatum_2_gpt4_result,
#         'Turbo': Ultimatum_2_turbo_result,
#     },
#     'Trust 1': {
#         'Human': Trust_1_human_result,
#         'GPT-4': Trust_1_gpt4_result,
#         'Turbo': Trust_1_turbo_result,
#     },
#     'Trust 3': {
#         'Human': Trust_3_human_result,
#         'GPT-4': Trust_3_gpt4_result,
#         'Turbo': Trust_3_turbo_result,
#     },
#     'Public Goods': {
#         'Human': Public_Goods_human_result,
#         'GPT-4': Public_Goods_gpt4_result,
#         'Turbo': Public_Goods_turbo_result,
#     },
#     'Bomb Risk': {
#         'Human': Bomb_Risk_human_result,
#         'GPT-4': Bomb_Risk_gpt4_result,
#         'Turbo': Bomb_Risk_turbo_result,
#     },
#     'Prisoner‘s Dilemma': {
#         'Human': Prisoner_Dilemma_human_result,
#         'GPT-4': Prisoner_Dilemma_gpt4_result,
#         'Turbo': Prisoner_Dilemma_turbo_result,
#     }
# }

# data = {
    
#     'Average': {
#         'Human': (0.39472, 0.20789, 0.39539),
#         'GPT-4': (0.45068, 0.20923, 0.34009),
#         'Turbo': (0.14191, 0.10243, 0.75566),
#     },
#     'Dictator': {
#         'Human': (0.39672, 0.20789, 0.39539),
#         'GPT-4': (0.45068, 0.20923, 0.34009),
#         'Turbo': (0.14191, 0.10243, 0.75566),
#     },
#     'Ultimatum 1': {
#         'Human': (0.39641, 0.20612, 0.39747),
#         'GPT-4': (0.65658, 0.34342, 0.0),
#         'Turbo': (0.35094, 0.19359, 0.45547),
#     },
#     'Ultimatum 2': {
#         'Human': (0.4103, 0.17962, 0.41008),
#         'GPT-4': (0.38777, 0.18544, 0.42679),
#         'Turbo': (0.23764, 0.12665, 0.63571),
#     },
#     'Trust 1': {
#         'Human': (0.42553, 0.1499, 0.42457),
#         'GPT-4': (0.24883, 0.0994, 0.65177),
#         'Turbo': (0.22511, 0.09314, 0.68175),
#     },
#     'Trust 3': {
#         'Human': (0.41412, 0.17279, 0.41309),
#         'GPT-4': (0.52801, 0.18393, 0.28806),
#         'Turbo': (0.52552, 0.17863, 0.29585),
#     },
#     'Public Goods': {
#         'Human': (0.42576, 0.14121, 0.43303),
#         'GPT-4': (0.63608, 0.20542, 0.1585),
#         'Turbo': (0.0, 1.0, 0.0),
#     },
#     'Bomb Risk': {
#         'Human': (0.43803, 0.12385, 0.43812),
#         'GPT-4': (0.66265, 0.18317, 0.15418),
#         'Turbo': (0.61836, 0.18062, 0.20102),
#     },
#     'Prisoner‘s Dilemma': {
#         'Human': (0.24761879117560937, 0.5047624176487813, 0.24761879117560937),
#         'GPT-4': (0.03760019738833667, 0.4593353072167, 0.5030644953949632),
#         'Turbo': (0.10528055268734268, 0.473974596618688, 0.42074485069396933),
#     }
# }

# # Transform the data into a format suitable for Altair
# records = []
# for scenario, scenario_data in data.items():
#     for entity, values in scenario_data.items():
#         records.append({
#             "scenario": scenario,
#             "entity": entity,
#             "Estimated More likely Human": values[0],
#             "Estimated Equally likely Human/AI": values[1],
#             "Estimated More Likely AI": values[2]
#         })

# # Create a DataFrame
# df = pd.DataFrame.from_records(records)

# # Melt the DataFrame to have category names and percentages in separate columns
# df_melted = df.melt(id_vars=['scenario', 'entity'], 
#                     value_vars=['Estimated More likely Human', 'Estimated Equally likely Human/AI', 'Estimated More Likely AI'],
#                     var_name='category', value_name='percentage')

# # Define the color scale
# color_scale = alt.Scale(domain=['Estimated More likely Human', 'Estimated Equally likely Human/AI', 'Estimated More Likely AI'],
#                         range=['#2ca02c', '#ff7f0e', '#d62728'])

# # Create the selection for the interactive highlight
# highlight = alt.selection_multi(on='mouseover',fields=['entity'], empty='none')
# # alt.selection_single(on='mouseover', fields=['category'], empty='none')

# # Define the base chart with a bar mark and the selection highlight
# base_chart = alt.Chart(df_melted).mark_bar().encode(
#     x=alt.X('sum(percentage)', stack="normalize", title='', axis=alt.Axis(format='.0%')),
#     y=alt.Y('entity:N', title='', sort=alt.EncodingSortField('entity', op='min', order='descending')),
#     color=alt.Color('category', scale=color_scale),
#     opacity=alt.condition(highlight, alt.value(1), alt.value(0.2)),
#     tooltip=[alt.Tooltip('scenario'), alt.Tooltip('entity'), alt.Tooltip('category'), alt.Tooltip('sum(percentage):Q', format='.2%')]
# ).properties(
#     width=450,  # Width of the individual charts
#     height=200
# ).add_selection(
#     highlight
# )

# # Facet the base chart into two rows
# faceted_chart = base_chart.facet(
#     facet=alt.Facet('scenario', title='', header=alt.Header(labelOrient='top', titleOrient='top')),
#     columns=3  # Display four graphs in each column
# )

# # Adjust spacing between the rows for better readability
# faceted_chart = faceted_chart.configure_facet(spacing=10)

# # Show the chart
# faceted_chart



# %% [markdown]
# ## Fig. 3: Distributions

# %% [markdown]
# ### Dictator Game
# 
# #### Human Data

# %%
binrange = (0, 100)
moves = []
with open('data/dictator.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    col2idx = {col: idx for idx, col in enumerate(header)}
    for row in reader:
        record = {col: row[idx] for col, idx in col2idx.items()}

        if record['Role'] != 'first': continue
        if int(record['Round']) > 1: continue
        if int(record['Total']) != 100: continue
        if record['move'] == 'None': continue
        if record['gameType'] != 'dictator': continue

        move = float(record['move'])
        if move < binrange[0] or \
            move > binrange[1]: continue
        
        moves.append(move)

df_dictator_human = choices_to_df(moves, 'Human')

# %% [markdown]
# #### Model Data

# %%
choices = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
df_dictator_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

choices = [25, 35, 70, 30, 20, 25, 40, 80, 30, 30, 40, 30, 30, 30, 30, 30, 40, 40, 30, 30, 40, 30, 60, 20, 40, 25, 30, 30, 30]
df_dictator_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

# %%
print(len(df_dictator_gpt4.dropna()))
print(len(df_dictator_turbo.dropna()))

# %%
records = json.load(open('records/dictator_wo_ex_2023_03_13-11_24_07_PM.json', 'r'))
choices = extract_choices(records)
print(', '.join([str(x) for x in choices]))

# %% [markdown]
# #### Plot

# %% [markdown]
# ### Ultimatum Game
# 
# #### Human Data

# %%
df = pd.read_csv('data/ultimatum_strategy.csv')
df = df[df['gameType'] == 'ultimatum_strategy']
df = df[df['Role'] == 'player']
df = df[df['Round'] == 1]
df = df[df['Total'] == 100]
df = df[df['move'] != 'None']
df['propose'] = df['move'].apply(lambda x: eval(x)[0])
df['accept'] = df['move'].apply(lambda x: eval(x)[1])
df = df[(df['propose'] >= 0) & (df['propose'] <= 100)]
df = df[(df['accept'] >= 0) & (df['accept'] <= 100)]
# df.head()

# %%
df_ultimatum_1_human = choices_to_df(list(df['propose']), 'Human')
df_ultimatum_2_human = choices_to_df(list(df['accept']), 'Human')

# %% [markdown]
# #### Model Data

# %%
choices = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
df_ultimatum_1_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

choices = [40, 40, 40, 30, 70, 70, 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 35, 50, 40, 70, 40, 60, 60, 70, 40, 50]
df_ultimatum_1_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

choices = [50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 50.0, 25.0, 50.0, 1.0, 1.0, 20.0, 50.0, 50.0, 50.0, 20.0, 50.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 20.0, 1.0] + [0, 1]
df_ultimatum_2_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

choices = [None, 50, 50, 50, 50, 30, None, None, 30, 33.33, 40, None, 50, 40, None, 1, 30, None, 10, 50, 30, 10, 30, None, 30, None, 10, 30, 30, 30]
df_ultimatum_2_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

# %%
print(len(df_ultimatum_1_gpt4.dropna()))
print(len(df_ultimatum_1_turbo.dropna()))

print(len(df_ultimatum_2_gpt4.dropna()))
print(len(df_ultimatum_2_turbo.dropna()))

# %% [markdown]
# #### Plot

# %% [markdown]
# #### Reproducibility

# %%
choices = [50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 50.0, 25.0, 50.0, 1.0, 1.0, 20.0, 50.0, 50.0, 50.0, 20.0, 50.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 20.0, 1.0] + [0, 1]
df_ultimatum_2_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4 (0314, 04.05)'))

# records = json.load(open('records/ultimatum_2_gpt4_2023_12_29-10_42_42_PM.json', 'r'))
# choices = extract_choices(records)
# df_ultimatum_2_gpt4_1229 = choices_to_df(choices, hue=str('ChatGPT-4 (0314, 12.29)'))

# records = json.load(open('records/ultimatum_2_gpt4_1106_2023_12_29-11_15_35_PM.json', 'r'))
# choices = extract_choices(records)
# df_ultimatum_2_gpt4_1106_1229 = choices_to_df(choices, hue=str('ChatGPT-4 (1106, 12.29)'))

# plot_facet(
#     df_list=[
#         df_ultimatum_2_gpt4,
#         df_ultimatum_2_gpt4_1229,
#         df_ultimatum_2_gpt4_1106_1229,
#     ],
#     binrange=(0, 100),
#     # binwidth=10,
#     stat='density',
#     x_label='Minimum proposal to accept ($)',
# )
# plt.savefig('figures/repro-ultimatum-respond.pdf', format='pdf', bbox_inches='tight')
# plt.show()

# %%
choices = [None, 50, 50, 50, 50, 30, None, None, 30, 33.33, 40, None, 50, 40, None, 1, 30, None, 10, 50, 30, 10, 30, None, 30, None, 10, 30, 30, 30]
df_ultimatum_2_turbo = choices_to_df(choices, hue=str('ChatGPT-3 (0301, 03.13)'))

# records = json.load(open('records/ultimatum_2_turbo_2023_12_29-10_43_41_PM.json', 'r'))
# choices = extract_choices(records)
# df_ultimatum_2_turbo_1229 = choices_to_df(choices, hue=str('ChatGPT-3 (0301, 12.29)'))

# records = json.load(open('records/ultimatum_2_turbo_1106_2023_12_29-11_57_30_PM.json', 'r'))
# choices = extract_choices(records)
# df_ultimatum_2_turbo_1106_1229 = choices_to_df(choices, hue=str('ChatGPT-3 (1106, 12.29)'))

# plot_facet(
#     df_list=[
#         df_ultimatum_2_turbo,
#         df_ultimatum_2_turbo_1229,
#         df_ultimatum_2_turbo_1106_1229,
#     ],
#     binrange=(0, 100),
#     # binwidth=10,
#     stat='density',
#     x_label='Minimum proposal to accept ($)',
# )
# # plt.savefig('figures/cmp-ultimatum-respond.pdf', format='pdf', bbox_inches='tight')
# plt.show()

# %% [markdown]
# ### Trust Game
# 
# #### Human Data

# %%
binrange = (0, 100)
moves_1 = []
moves_2 = defaultdict(list)
with open('data/trust_investment.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    col2idx = {col: idx for idx, col in enumerate(header)}
    for row in reader:
        record = {col: row[idx] for col, idx in col2idx.items()}

        # if record['Role'] != 'first': continue
        if int(record['Round']) > 1: continue
        # if int(record['Total']) != 100: continue
        if record['move'] == 'None': continue
        if record['gameType'] != 'trust_investment': continue

        if record['Role'] == 'first':
            move = float(record['move'])
            if move < binrange[0] or \
                move > binrange[1]: continue
            moves_1.append(move)
        elif record['Role'] == 'second':
            inv, ret = eval(record['roundResult'])
            if ret < 0 or \
                ret > inv * 3: continue
            moves_2[inv].append(ret)
        else: continue

df_trust_1_human = choices_to_df(moves_1, 'Human')
df_trust_2_human = choices_to_df(moves_2[10], 'Human')
df_trust_3_human = choices_to_df(moves_2[50], 'Human')
df_trust_4_human = choices_to_df(moves_2[100], 'Human')

# %% [markdown]
# #### Model Data

# %%
choices = [50.0, 50.0, 40.0, 30.0, 50.0, 50.0, 40.0, 50.0, 50.0, 50.0, 50.0, 50.0, 30.0, 30.0, 50.0, 50.0, 50.0, 40.0, 40.0, 50.0, 50.0, 50.0, 50.0, 40.0, 50.0, 50.0, 50.0, 50.0] 
df_trust_1_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

choices = [50.0, 50.0, 30.0, 30.0, 30.0, 60.0, 50.0, 40.0, 20.0, 20.0, 50.0, 40.0, 30.0, 20.0, 30.0, 20.0, 30.0, 60.0, 50.0, 30.0, 50.0, 20.0, 20.0, 30.0, 50.0, 30.0, 30.0, 50.0, 40.0] + [30]
df_trust_1_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

choices = [20.0, 20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 15.0, 20.0, 15.0, 15.0, 15.0, 15.0, 15.0, 20.0, 20.0, 15.0]
df_trust_2_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

choices = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 25.0, 30.0, 30.0, 20.0, 25.0, 30.0, 20.0, 20.0, 18.0] + [20, 20, 20, 25, 25, 25, 30]
df_trust_2_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

choices = [100.0, 75.0, 75.0, 75.0, 75.0, 75.0, 100.0, 75.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 75.0, 100.0, 75.0, 75.0, 75.0, 100.0, 100.0, 100.0, 75.0, 100.0, 100.0, 100.0, 100.0, 75.0, 100.0, 75.0]
df_trust_3_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

choices = [150.0, 100.0, 150.0, 150.0, 50.0, 150.0, 100.0, 150.0, 100.0, 100.0, 100.0, 150.0] + [100, 100, 100, 100, 100, 100, 100, 100]
df_trust_3_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

choices = [200.0, 200.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 200.0, 200.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]
df_trust_4_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

choices = [225.0, 225.0, 300.0, 300.0, 220.0, 300.0, 250.0] + [200, 200, 250, 200, 200]
df_trust_4_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

# %%
records = json.load(open('records/trust_2_gpt4_2023_04_07-11_46_45_PM.json', 'r'))
choices = extract_choices(records)
print(', '.join([str(c) for c in choices]))

# %%
records = json.load(open('records/trust_4_gpt4_2023_04_08-12_24_56_AM.json', 'r'))
choices = extract_choices(records)
print(', '.join([str(c) for c in choices]))

# %% [markdown]
# #### Plot

# %% [markdown]
# ### Public Goods
# 
# #### Human Data

# %%
df = pd.read_csv('data/public_goods_linear_water.csv')
df = df[df['Role'] == 'contributor']
df = df[df['Round'] <= 3]
df = df[df['Total'] == 20]
df = df[df['groupSize'] == 4]
df = df[df['move'] != None]
df = df[(df['move'] >= 0) & (df['move'] <= 20)]
df = df[df['gameType'] == 'public_goods_linear_water']

# %%
round_1 = df[df['Round'] == 1]['move']
round_2 = df[df['Round'] == 2]['move']
round_3 = df[df['Round'] == 3]['move']
print(len(round_1), len(round_2), len(round_3))
df_PG_human = pd.DataFrame({
    'choices': list(round_1)
})
df_PG_human['hue'] = 'Human'
df_PG_human

# %% [markdown]
# ##### Payoffs

# %% [markdown]
# #### Model Data

# %%
file_names = [
    # 'records/PG_basic_turbo_2023_05_09-02_49_09_AM.json',
    # 'records/PG_basic_turbo_loss_2023_05_09-03_59_49_AM.json'
    'records/PG_basic_gpt4_2023_05_09-11_15_42_PM.json',
    'records/PG_basic_gpt4_loss_2023_05_09-10_44_38_PM.json',
]

choices = []
for file_name in file_names:
    with open(file_name, 'r') as f:
        choices += json.load(f)['choices']
choices_baseline = choices

choices = [tuple(x)[0] for x in choices]
df_PG_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))
df_PG_turbo.head()



# %%
file_names = [
    # 'records/PG_basic_turbo_2023_05_09-02_49_09_AM.json',
    # 'records/PG_basic_turbo_loss_2023_05_09-03_59_49_AM.json'
    'records/PG_basic_gpt4_2023_05_09-11_15_42_PM.json',
    'records/PG_basic_gpt4_loss_2023_05_09-10_44_38_PM.json',
]

choices = []
for file_name in file_names:
    with open(file_name, 'r') as f:
        choices += json.load(f)['choices']
choices_baseline = choices

choices = [tuple(x)[0] for x in choices]
df_PG_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))
df_PG_gpt4.head()

# %% [markdown]
# #### Plot

# %% [markdown]
# ### Bomb Risk Game
# 
# 1 safe, 0 bomb
# 
# #### Human Data

# %%
df = pd.read_csv('data/bomb_risk.csv')
df = df[df['Role'] == 'player']
df = df[df['gameType'] == 'bomb_risk']
df.sort_values(by=['UserID', 'Round'])

prefix_to_choices_human = defaultdict(list)
prefix_to_IPW = defaultdict(list)
prev_user = None
prev_move = None
prefix = ''
bad_user = False
for _, row in df.iterrows():
    if bad_user: continue
    if row['UserID'] != prev_user:
        prev_user = row['UserID']
        prefix = ''
        bad_user = False

    move = row['move']
    if move < 0 or move > 100:
        bad_users = True
        continue
    prefix_to_choices_human[prefix].append(move)

    if len(prefix) == 0:
        prefix_to_IPW[prefix].append(1)
    elif prefix[-1] == '1':
        prev_move = min(prev_move, 98)
        prefix_to_IPW[prefix].append(1./(100 - prev_move))
    elif prefix[-1] == '0':
        prev_move = max(prev_move, 1)
        prefix_to_IPW[prefix].append(1./(prev_move))
    else: assert False
    
    prev_move = move

    prefix += '1' if row['roundResult'] == 'SAFE' else '0'

# %% [markdown]
# #### Model Data

# %%
prefix_to_choices_model = defaultdict(lambda : defaultdict(list))
for model in ['ChatGPT-4', 'ChatGPT-3']:
    if model == 'ChatGPT-4':
        file_names = [
            'bomb_gpt4_2023_05_15-12_13_51_AM.json'
        ]
    elif model == 'ChatGPT-3':
        file_names = [
            'bomb_turbo_2023_05_14-10_45_50_PM.json'
        ]

    choices = []
    scenarios = []
    for file_name in file_names:
        with open(os.path.join('records', file_name), 'r') as f:
            records = json.load(f)
            choices += records['choices']
            scenarios += records['scenarios']

    assert len(scenarios) == len(choices)
    print('loaded %i valid records' % len(scenarios))

    prefix_to_choice = defaultdict(list)
    prefix_to_result = defaultdict(list)
    prefix_to_pattern = defaultdict(Counter)
    wrong_sum = 0
    for scenarios_tmp, choices_tmp in zip(scenarios, choices):

        result = 0
        for i, scenario in enumerate(scenarios_tmp):
            prefix = tuple(scenarios_tmp[:i])
            prefix = ''.join([str(x) for x in prefix])
            choice = choices_tmp[i]
            
            prefix_to_choice[prefix].append(choice)
            prefix_to_pattern[prefix][tuple(choices_tmp[:-1])] += 1

            prefix = tuple(scenarios_tmp[:i+1])
            if scenario == 1:
                result += choice
            prefix_to_result[prefix].append(result)

    print('# of wrong sum:', wrong_sum)
    print('# of correct sum:', len(scenarios) - wrong_sum)

    prefix_to_choices_model[model] = prefix_to_choice

# %% [markdown]
# #### Plot

# %% [markdown]
# ### Prisoner Dilemma
# 
# #### Human Data

# %%
df = pd.read_csv('data/push_pull.csv')
df = df[df['gameType'] == 'push_pull']
df = df[df['Role'] == 'player']
df = df[(df['move'] == 0) | (df['move'] == 1)]
# df = df[df['Round'] <= 2]
df = df[df['groupSize'] == 2]

counter = -1
playIDs = []
otherMoves = []
for i, row in df.iterrows():
    if row['Round'] == 1:
        counter += 1
    playIDs.append(counter)
    roundResult = eval(row['roundResult'])
    roundResult.remove(row['move'])
    otherMoves.append(roundResult[0])
df['playID'] = playIDs
df['otherMove'] = otherMoves

# %%
n_coo_human = 36269
n_def_human = 44114
r_coo_human = n_coo_human / (n_coo_human + n_def_human)
r_def_human = n_def_human / (n_coo_human + n_def_human)
print(r_coo_human, r_def_human)

# %%
print(counter)

# %% [markdown]
# #### Model Data

# %%
n_coo_gpt4 = 29 + 0 + 0 + 26
n_def_gpt4 = 0 + 1 + 1 + 3
n_coo_turbo = 21 + 3 + 7 + 15
n_def_turbo = 3 + 3 + 4 + 4
r_coo_gpt4 = n_coo_gpt4 / (n_coo_gpt4 + n_def_gpt4)
r_def_gpt4 = n_def_gpt4 / (n_coo_gpt4 + n_def_gpt4)
r_coo_turbo = n_coo_turbo / (n_coo_turbo + n_def_turbo)
r_def_turbo = n_def_turbo / (n_coo_turbo + n_def_turbo)
print(r_coo_gpt4, r_def_gpt4)
print(r_coo_turbo, r_def_turbo)

# %%
file_names = [
    'records/PD_gpt4_two_rounds_push_2023_05_10-10_04_33_PM.json',
    'records/PD_gpt4_two_rounds_pull_2023_05_08-08_57_08_PM.json'
]

choices = []
for file_name in file_names:
    with open(file_name, 'r') as f:
        choices += json.load(f)['choices']
choices_baseline = choices

choices = [tuple(x)[0] for x in choices]
df_PG_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))
df_PG_gpt4.head()

# %%
file_names = [
    'records/PD_turbo_two_rounds_push_2023_05_08-06_03_40_PM.json',
    'records/PD_turbo_two_rounds_pull_2023_05_08-09_23_13_PM.json',
]

choices = []
for file_name in file_names:
    with open(file_name, 'r') as f:
        choices += json.load(f)['choices']
choices_baseline = choices

choices = [tuple(x)[0] for x in choices]
df_PG_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))
df_PG_turbo.head()

# %% [markdown]
# #### Prisoner's Dilemma

# %%
r_coo = r_coo_human
r_def = r_def_human
S = [r_coo * 400 + r_def * 000, r_coo * 700 + r_def * 300]
P = [r_coo * 400 + r_def * 700, r_coo * 000 + r_def * 300]
k = {}
for model in ['Human', 'ChatGPT-4', 'ChatGPT-3']:    
    if model == 'Human':
        n_coo = n_coo_human
        n_def = n_def_human
    elif model == 'ChatGPT-4':
        n_coo = n_coo_gpt4
        n_def = n_def_gpt4
    elif model == 'ChatGPT-3':
        n_coo = n_coo_turbo
        n_def = n_def_turbo
    k[model] = [0] * n_coo + [1] * n_def
    print(model)
    # # estimate_beta(S, P, k[model])
    # estimate_beta(S, P, k[model], r=.5)

# %% [markdown]
# ### Occupation

# %% [markdown]
# ### Investment (Trust 2-4)

# %% [markdown]
# #### Session 2

# %% [markdown]
# ### Trust Game

# %% [markdown]
# ## Fig. 2: Turing Test
# 
# Results may vary due to randomness in the simulation.

# %%
def simulate_Turing(samples_0, samples_1, n_bin=10, lim_a=0, lim_b=100, n_draw=100000):
    hist_0 = np.histogram(samples_0, bins=n_bin, range=(lim_a, lim_b))[0] / len(samples_0)
    n_wins = 0
    n_ties = 0
    for _ in tqdm(range(n_draw)):
        try:
            sample_0 = np.random.choice(samples_0)
            sample_1 = np.random.choice(samples_1)
            idx_0 = min(math.floor((sample_0 - lim_a) / (lim_b - lim_a) * n_bin), n_bin-1)
            idx_1 = min(math.floor((sample_1 - lim_a) / (lim_b - lim_a) * n_bin), n_bin-1)
            if hist_0[idx_1] > hist_0[idx_0]:
                n_wins += 1
            elif hist_0[idx_1] == hist_0[idx_0]:
                n_ties += 1
        except:
            continue
    return n_wins / n_draw, n_ties / n_draw, (n_draw - n_wins - n_ties) / n_draw

# %%


# %%
print('human', simulate_Turing(df_dictator_human['choices'], df_dictator_human['choices']))
print('gpt4', simulate_Turing(df_dictator_human['choices'], df_dictator_gpt4['choices']))
print('turbo', simulate_Turing(df_dictator_human['choices'], df_dictator_turbo['choices']))

# %%
print('human', simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_human['choices']))
print('gpt4', simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_gpt4['choices']))
print('turbo', simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_turbo['choices']))

# %%
print('human', simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_human['choices']))
print('gpt4', simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_gpt4['choices']))
print('turbo', simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_turbo['choices']))

# %%
print('human', simulate_Turing(df_trust_1_human['choices'], df_trust_1_human['choices']))
print('gpt4', simulate_Turing(df_trust_1_human['choices'], df_trust_1_gpt4['choices']))
print('turbo', simulate_Turing(df_trust_1_human['choices'], df_trust_1_turbo['choices']))

# %%
print('human', simulate_Turing(df_trust_3_human['choices'], df_trust_3_human['choices'], lim_b=150))
print('gpt4', simulate_Turing(df_trust_3_human['choices'], df_trust_3_gpt4['choices'], lim_b=150))
print('turbo', simulate_Turing(df_trust_3_human['choices'], df_trust_3_turbo['choices'], lim_b=150))

# %%
# # Convert to numeric, coercing errors (invalid parsing will be set as NaN)
# samples_0 = pd.to_numeric(df_PG_humapt4['choices'], errors='coerce')n['choices'], errors='coerce')
# samples_1 = pd.to_numeric(df_PG_g

# %%
# df_PG_turbo['choices'].fillna(0, inplace=True)


# %%
# df_PG_turbo['choices'] = pd.to_numeric(df_PG_turbo['choices'], errors='coerce')

# %%
print (np.unique(df_PG_turbo['choices']))

# %%
print('human', simulate_Turing(df_PG_human['choices'], df_PG_human['choices'], lim_b=20))
print('gpt4', simulate_Turing(df_PG_human['choices'], df_PG_gpt4['choices'], lim_b=20))
print('gpt4', simulate_Turing(df_PG_turbo['choices'], df_PG_turbo['choices'], lim_b=20))

# %%
print('human', simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_human['']))
print('gpt4', simulate_Turing(prefix_to_choices_human[''],prefix_to_choices_model['ChatGPT-4']['']))
print('turbo', simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_model['ChatGPT-3']['']))

# %%
print('human', r_coo_human * r_def_human, r_coo_human * r_coo_human + r_def_human * r_def_human, r_def_human * r_coo_human)
print('gpt4', r_coo_human * r_def_gpt4, r_coo_human * r_coo_gpt4 + r_def_human * r_def_gpt4, r_def_human * r_coo_gpt4)
print('turbo', r_coo_human * r_def_turbo, r_coo_human * r_coo_turbo + r_def_human * r_def_turbo, r_def_human * r_coo_turbo)

# %%
# data = {

#     'Average': {
#         'Human': (0.39472, 0.20789, 0.39539),
#         'GPT-4': (0.45068, 0.20923, 0.34009),
#         'Turbo': (0.14191, 0.10243, 0.75566),
#     },
#     'Dictator': {
#         'Human': (0.39672, 0.20789, 0.39539),
#         'GPT-4': (0.45068, 0.20923, 0.34009),
#         'Turbo': (0.14191, 0.10243, 0.75566),
#     },
#     'Ultimatum 1': {
#         'Human': (0.39641, 0.20612, 0.39747),
#         'GPT-4': (0.65658, 0.34342, 0.0),
#         'Turbo': (0.35094, 0.19359, 0.45547),
#     },
#     'Ultimatum 2': {
#         'Human': (0.4103, 0.17962, 0.41008),
#         'GPT-4': (0.38777, 0.18544, 0.42679),
#         'Turbo': (0.23764, 0.12665, 0.63571),
#     },
#     'Trust 1': {
#         'Human': (0.42553, 0.1499, 0.42457),
#         'GPT-4': (0.24883, 0.0994, 0.65177),
#         'Turbo': (0.22511, 0.09314, 0.68175),
#     },
#     'Trust 3': {
#         'Human': (0.41412, 0.17279, 0.41309),
#         'GPT-4': (0.52801, 0.18393, 0.28806),
#         'Turbo': (0.52552, 0.17863, 0.29585),
#     },
#     'Public Goods': {
#         'Human': (0.42576, 0.14121, 0.43303),
#         'GPT-4': (0.63608, 0.20542, 0.1585),
#         'Turbo': (0.0, 1.0, 0.0),
#     },
#     'Bomb Risk': {
#         'Human': (0.43803, 0.12385, 0.43812),
#         'GPT-4': (0.66265, 0.18317, 0.15418),
#         'Turbo': (0.61836, 0.18062, 0.20102),
#     },
#     'Prisoner‘s Dilemma': {
#         'Human': (0.24761879117560937, 0.5047624176487813, 0.24761879117560937),
#         'GPT-4': (0.03760019738833667, 0.4593353072167, 0.5030644953949632),
#         'Turbo': (0.10528055268734268, 0.473974596618688, 0.42074485069396933),
#     }
# }

    Dictator_human_result = simulate_Turing(df_dictator_human['choices'], df_dictator_human['choices'])
    Dictator_gpt4_result = simulate_Turing(df_dictator_human['choices'], df_dictator_gpt4['choices'])
    Dictator_turbo_result = simulate_Turing(df_dictator_human['choices'], df_dictator_turbo['choices'])

    Ultimatum_1_human_result = simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_human['choices'])
    Ultimatum_1_gpt4_result = simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_gpt4['choices'])
    Ultimatum_1_turbo_result = simulate_Turing(df_ultimatum_1_human['choices'], df_ultimatum_1_turbo['choices'])

    Ultimatum_2_human_result = simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_human['choices'])
    Ultimatum_2_gpt4_result = simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_gpt4['choices'])
    Ultimatum_2_turbo_result = simulate_Turing(df_ultimatum_2_human['choices'], df_ultimatum_2_turbo['choices'])

    Trust_1_human_result = simulate_Turing(df_trust_1_human['choices'], df_trust_1_human['choices'])
    Trust_1_gpt4_result = simulate_Turing(df_trust_1_human['choices'], df_trust_1_gpt4['choices'])
    Trust_1_turbo_result = simulate_Turing(df_trust_1_human['choices'], df_trust_1_turbo['choices'])

    Trust_3_human_result = simulate_Turing(df_trust_3_human['choices'], df_trust_3_human['choices'], lim_b=150)
    Trust_3_gpt4_result = simulate_Turing(df_trust_3_human['choices'], df_trust_3_gpt4['choices'], lim_b=150)
    Trust_3_turbo_result = simulate_Turing(df_trust_3_human['choices'], df_trust_3_turbo['choices'], lim_b=150)

    Public_Goods_human_result = simulate_Turing(df_PG_human['choices'], df_PG_human['choices'], lim_b=20)
    Public_Goods_gpt4_result = simulate_Turing(df_PG_human['choices'], df_PG_gpt4['choices'], lim_b=20)
    Public_Goods_turbo_result = simulate_Turing(df_PG_turbo['choices'], df_PG_turbo['choices'], lim_b=20)

    Bomb_Risk_human_result = simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_human[''])
    Bomb_Risk_gpt4_result = simulate_Turing(prefix_to_choices_human[''],prefix_to_choices_model['ChatGPT-4'][''])
    Bomb_Risk_turbo_result = simulate_Turing(prefix_to_choices_human[''], prefix_to_choices_model['ChatGPT-3'][''])

    Prisoner_Dilemma_human_result = r_coo_human * r_def_human, r_coo_human * r_coo_human + r_def_human * r_def_human, r_def_human * r_coo_human
    Prisoner_Dilemma_gpt4_result = r_coo_human * r_def_gpt4, r_coo_human * r_coo_gpt4 + r_def_human * r_def_gpt4, r_def_human * r_coo_gpt4
    Prisoner_Dilemma_turbo_result = r_coo_human * r_def_turbo, r_coo_human * r_coo_turbo + r_def_human * r_def_turbo, r_def_human * r_coo_turbo



    # %%

    Average_human_result = tuple(sum(results) / 8 for results in zip(Dictator_human_result, Ultimatum_1_human_result, Ultimatum_2_human_result, Trust_1_human_result, Trust_3_human_result, Public_Goods_human_result, Bomb_Risk_human_result, Prisoner_Dilemma_human_result))
    Average_gpt4_result = tuple(sum(results) / 8 for results in zip(Dictator_gpt4_result, Ultimatum_1_gpt4_result, Ultimatum_2_gpt4_result, Trust_1_gpt4_result, Trust_3_gpt4_result, Public_Goods_gpt4_result, Bomb_Risk_gpt4_result, Prisoner_Dilemma_gpt4_result))
    Average_turbo_result = tuple(sum(results) / 8 for results in zip(Dictator_turbo_result, Ultimatum_1_turbo_result, Ultimatum_2_turbo_result, Trust_1_turbo_result, Trust_3_turbo_result, Public_Goods_turbo_result, Bomb_Risk_turbo_result, Prisoner_Dilemma_turbo_result))

    data = {
        'Average': {
            'Human': Average_human_result,
            'GPT-4': Average_gpt4_result,
            'GPT-3': Average_turbo_result,
        },
        'Dictator': {
            'Human': Dictator_human_result,
            'GPT-4': Dictator_gpt4_result,
            'GPT-3': Dictator_turbo_result,
        },
        'Ultimatum 1': {
            'Human': Ultimatum_1_human_result,
            'GPT-4': Ultimatum_1_gpt4_result,
            'GPT-3': Ultimatum_1_turbo_result,
        },
        'Ultimatum 2': {
            'Human': Ultimatum_2_human_result,
            'GPT-4': Ultimatum_2_gpt4_result,
            'GPT-3': Ultimatum_2_turbo_result,
        },
        'Trust 1': {
            'Human': Trust_1_human_result,
            'GPT-4': Trust_1_gpt4_result,
            'GPT-3': Trust_1_turbo_result,
        },
        'Trust 3': {
            'Human': Trust_3_human_result,
            'GPT-4': Trust_3_gpt4_result,
            'GPT-3': Trust_3_turbo_result,
        },
        'Public Goods': {
            'Human': Public_Goods_human_result,
            'GPT-4': Public_Goods_gpt4_result,
            'GPT-3': Public_Goods_turbo_result,
        },
        'Bomb Risk': {
            'Human': Bomb_Risk_human_result,
            'GPT-4': Bomb_Risk_gpt4_result,
            'GPT-3': Bomb_Risk_turbo_result,
        },
        'Prisoner‘s Dilemma': {
            'Human': Prisoner_Dilemma_human_result,
            'GPT-4': Prisoner_Dilemma_gpt4_result,
            'GPT-3': Prisoner_Dilemma_turbo_result,
        }
    }

    # %%
    # Transform the data into a format suitable for Altair
    records = []
    for scenario, scenario_data in data.items():
        for entity, values in scenario_data.items():
            records.append({
                "scenario": scenario,
                "entity": entity,
                "Estimated More likely Human": values[0],
                "Estimated Equally likely Human/AI": values[1],
                "Estimated More Likely AI": values[2]
            })

    # Create a DataFrame
    df = pd.DataFrame.from_records(records)

    # Melt the DataFrame to have category names and percentages in separate columns
    df_melted = df.melt(id_vars=['scenario', 'entity'], 
                        value_vars=['Estimated More likely Human', 'Estimated Equally likely Human/AI', 'Estimated More Likely AI'],
                        var_name='category', value_name='percentage')

    # Define the color scale
    color_scale = alt.Scale(domain=['Estimated More likely Human', 'Estimated Equally likely Human/AI', 'Estimated More Likely AI'],
                            range=['#2ca02c', '#ff7f0e', '#d62728'])

    # Create the selection for the interactive highlight
    highlight = alt.selection_multi(on='click',fields=['entity'], empty='none')
    # alt.selection_single(on='mouseover', fields=['category'], empty='none')

    # Define the base chart with a bar mark and the selection highlight
    base_chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('sum(percentage)', stack="normalize", title='', axis=alt.Axis(format='.0%')),
        y=alt.Y('entity:N', title='', sort=alt.EncodingSortField('entity', op='min', order='descending')),
        color=alt.Color('category', scale=color_scale),
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.6)),
        tooltip=[alt.Tooltip('scenario'), alt.Tooltip('entity'), alt.Tooltip('category'), alt.Tooltip('sum(percentage):Q', format='.2%')]
    ).properties(
        width=450,  # Width of the individual charts
        height=200
    ).add_selection(
        highlight,
    )

    # Facet the base chart into two rows
    faceted_chart = base_chart.facet(
        facet=alt.Facet('scenario', title='', header=alt.Header(labelOrient='top', titleOrient='top')),
        columns=3  # Display four graphs in each column
    )

    # Adjust spacing between the rows for better readability
    faceted_chart = faceted_chart.configure_facet(spacing=10)

    faceted_chart


