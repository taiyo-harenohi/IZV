#!/usr/bin/env python3.11
# coding=utf-8
"""
IZV cast1 projektu
Autor: Nikola Machalkova (xmacha80)
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import io

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz


# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str) -> pd.DataFrame:
    """Load data from .csv files into DataFrame

    Keyword arguments:
    filename -- path to the file
    """
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    data_frames = []

    with zipfile.ZipFile(filename, 'r') as outer_folder:
        for inner_folder in outer_folder.infolist():
            with outer_folder.open(inner_folder) as inner_zip_folder:
                with zipfile.ZipFile(io.BytesIO(inner_zip_folder.read())) as inner_zip:
                    for csv_files in inner_zip.namelist():
                        if inner_zip.getinfo(csv_files).file_size > 0:
                            for region_name, region_code in regions.items():
                                if region_code in csv_files.replace(".csv", ""):
                                    df = pd.read_csv(inner_zip.open(csv_files),
                                                     encoding='cp1250',
                                                     on_bad_lines='skip', 
                                                     header=None,
                                                     names=headers, 
                                                     low_memory=False,
                                                     sep=';')
                                    df.insert(0, "region", region_name)
                                    data_frames.append(df)

    data_frame = pd.concat(data_frames, ignore_index=True)
    return data_frame


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Parse data of the DataFrame

    Keyword arguments:
    df -- DataFrame that needs to be parsed
    verbose -- boolean for writing on the standard output the deep size
    """
    data_frame = df
    if verbose:
        print(f"orig_size = {data_frame.memory_usage(deep=True).sum() / (10 ** 6):.1f} MB")

    data_frame.insert(0, "date", pd.to_datetime(data_frame["p2a"]))

    duplicate_mask = data_frame.drop_duplicates(subset=['p1'])

    float_columns = ["p37", "a", "b", "d", "e", "f", "g", "o", "n", "r", "q", "s"]
    category_columns = ["p1", "p6", "p8", "p12", "p16", "p17", "p22", "p27", "p32", "p33d", "p33e", "p35", "p44", "p45a", "p47", "p48a", "p52", "h", "i", "k", "p", "t"]
    int_columns = ["p54a", "p58", "l", "p10", "p2b", "p11"]


    for column_name, column_data in data_frame.items():
        if column_name != "region":
            if column_name in float_columns or column_name in int_columns:
                data_frame[column_name] = pd.to_numeric(data_frame[column_name], errors='coerce')
            if column_name in category_columns:
                data_frame[column_name] = data_frame[column_name].astype('category')
            
    if verbose:
        print(f"new_size = {data_frame.memory_usage(deep=True).sum() / (10 ** 6):.1f} MB")
        
    return data_frame

 
# Ukol 3: počty nehod podle stavu řidiče
def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """Generate graphs for the state of the driver at the time of the accident

    Keyword arguments:
    df -- DataFrame with information
    fig_location -- where will be the figure saved at
    show_figure -- boolean for showing figure on screen
    """
    df['Stav řidiče:'] = df.loc[~df['p57'].isin([0, 1, 2, 3]), 'p57']    
    filtered_df = df[["Stav řidiče:", "region"]]
    mapping = {7: 'invalida', 6: 'nemoc, úraz apod.', 5: 'pod vlivem alkoholu 1 % a více', 4: 'pod vlivem alkoholu do 0,99 %', 9: 'sebevražda', 8: 'řidič při jízdě zemřel'}
    filtered_df = filtered_df.replace(mapping)
    filtered_df['region'] = filtered_df['region'].astype('category')
    filtered_df['region'] = filtered_df['region'].sort_values()

    custom_palette = sns.color_palette("husl", n_colors=len(filtered_df['region'].unique()))

    g = sns.FacetGrid(filtered_df, col="Stav řidiče:", col_wrap=2, height=5, sharex=True, sharey=False, col_order=list(mapping.values()))
    g.map(sns.histplot, 'region', data=filtered_df, kde=False, hue='region', palette=custom_palette, legend=False, edgecolor='white', linewidth=2)

    # visual changes
    g.set_axis_labels("Kraj", "Počet nehod")
    g.set_titles("Stav řidiče: {col_name}", fontsize=16)

    for ax in g.axes.flatten():
        ax.margins(x=0.1)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        for y_value in ax.get_yticks():
            ax.axhline(y=y_value, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout(pad=2)

    plt.savefig(fig_location)
    if show_figure:
        plt.show()
    # clear the figure
    plt.clf()


# Ukol4: alkohol v jednotlivých hodinách
def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """Generate graphs if alcohol was present or not 

    Keyword arguments:
    df -- DataFrame with information
    fig_location -- where will be the figure saved at
    show_figure -- boolean for showing figure on screen
    """
    df["p2b"] = (df["p2b"] // 100) % 24
    df = df[df["p2b"].between(0, 23)].copy()
    df_yes_alcohol = df[df['p11'] >= 3].copy()
    df_no_alcohol = df[(df['p11'] == 1) | (df['p11'] == 2)].copy()
    df_yes_alcohol['Alkohol'] = 'Ano'
    df_no_alcohol['Alkohol'] = 'Ne'
    df_alcohol = pd.concat([df_yes_alcohol, df_no_alcohol])

    df_alcohol['p2b'] = df_alcohol['p2b'].astype(int)
    selected_regions = ['JHM', 'MSK', 'OLK', 'ZLK']
    df_alcohol = df_alcohol[df_alcohol['region'].isin(selected_regions)]
    df_alcohol.sort_values(by='p2b', inplace=True)

    df_counts = df_alcohol.groupby(['region', 'p2b', 'Alkohol']).size().reset_index(name='Count')

    g = sns.catplot(data=df_counts, x='p2b', y='Count', hue='Alkohol', col='region',
                    kind='bar', palette=['pink', 'lightblue'], hue_order=['Ne', 'Ano'],
                    col_order=['JHM', 'MSK', 'OLK', 'ZLK'], height=4, col_wrap=2, aspect=1.5, dodge=True)

    # visual changes
    g.set_axis_labels("Hodina", "Počet nehod")
    g.set_titles("Kraj: {col_name}", fontsize=16)

    for ax in g.axes.flatten():
        y_ticks = range(0, df_counts['Count'].max() + 1, 500)
        ax.set_yticks(y_ticks)
        ax.set_ylim(0, 3000)
        ax.margins(x=0.1)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        for y_value in ax.get_yticks():
            ax.axhline(y=y_value, color='gray', linestyle='--', linewidth=1)
    plt.tight_layout(pad=4)

    plt.savefig(fig_location)
    if show_figure:
        plt.show()

    plt.clf()


# Ukol 5: Zavinění nehody v čase
def plot_fault(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """Generate graphs of who was at fault of the accident

    Keyword arguments:
    df -- DataFrame with information
    fig_location -- where will be the figure saved at
    show_figure -- boolean for showing figure on screen
    """
    selected_regions = ['JHM', 'MSK', 'OLK', 'ZLK']
    df_accident = df[(df['region'].isin(selected_regions)) & (df['p10'].between(1, 4))].copy()
    mapping = {3: 'Chodcem', 4: 'Zvířetem', 1: 'Řidičem motorového vozidla', 2: 'Řidičem nemotorového vozidla'}
    df_accident['cause'] = df_accident['p10'].replace(mapping)

    df_accident['year'] = df_accident['date'].dt.year
    df_accident['month'] = df_accident['date'].dt.month
    df_accident['month_year'] = df_accident['date'].dt.to_period("M")

    df_pivot = df_accident.groupby(['region', 'month_year', 'cause']).size().reset_index(name='n_causes')
    df_stacked = df_pivot.pivot_table(values='n_causes', index=['region', 'month_year'], columns='cause', aggfunc='sum', fill_value=0).reset_index()
    
    df_melted = df_stacked.melt(id_vars=['region', 'month_year'], var_name='cause', value_name='n_causes')
    df_melted['month_year'] = pd.to_datetime(df_melted['month_year'].dt.to_timestamp())

    unique_causes = df_melted['cause'].unique()
    hls_palette = sns.color_palette("hls", n_colors=len(unique_causes))
    cause_color_mapping = dict(zip(unique_causes, hls_palette))
    df_melted['cause_color'] = df_melted['cause'].map(cause_color_mapping)

    g = sns.FacetGrid(df_melted, col='region', col_wrap=2, height=5, hue='cause', palette=cause_color_mapping)
    g.map(sns.lineplot, 'month_year', 'n_causes')
    plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2023-01-01'))

    # visual changes
    g.set_axis_labels("Datum", "Počet nehod")
    g.set_titles("Kraj: {col_name}", fontsize=16)
    plt.legend()

    # Adjust layout
    plt.tight_layout(pad=2)
    
    plt.savefig(fig_location)
    if show_figure:
        plt.show()

    plt.clf()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.
    df = load_data("data/data.zip")
    df2 = parse_data(df, True)

    plot_state(df2, "01_state.png")
    plot_alcohol(df2, "02_alcohol.png", True)
    plot_fault(df2, "03_fault.png")


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
