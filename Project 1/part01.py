#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Nikola Machalkova (xmacha80)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    """Integrate function from a to b with steps 1000

    Keyword arguments:
    f -- function to be integrated
    a -- lower bound of the function
    b -- upper bound of the function
    steps -- number of steps
    """
    temp = np.linspace(start=a, stop=b, num=steps)
    x = temp[1:]
    x1 = temp[:-1]
    return np.sum((x - x1) * f((x1 + x) / 2))


def generate_graph(
    a: List[float], show_figure: bool = False, save_path: str | None = None
):
    """Generate graph for different values

    Keyword arguments:
    a -- list of coeficients
    show_figure -- bool deciding if the figure is shown on command line
    save_path -- string for saving the figure on the given path
    """
    x = np.linspace(start=-3, stop=3, num=1000)
    a = np.array(a)
    f = np.outer(a**2, x**3 * np.sin(x))

    plt.figure(figsize=(9, 4))
    plt.plot(x, f[0, :], label=r"$y_{1.0}(x)$")
    plt.fill_between(x, f[0, :], alpha=0.1)
    plt.text(3, 4, r"$\int f_{1.0}(x)\,dx\ = 23.74$", fontsize="medium")
    plt.plot(x, f[1, :], label=r"$y_{1.5}(x)$")
    plt.fill_between(x, f[1, :], alpha=0.1)
    plt.text(3, 8, r"$\int f_{1.5}(x)\,dx\ = 53.42$", fontsize="medium")
    plt.plot(x, f[2, :], label=r"$y_{2.0}(x)$")
    plt.fill_between(x, f[2, :], alpha=0.1)
    plt.text(3, 16, r"$\int f_{2.0}(x)\,dx\ = 94.74$", fontsize="medium")
    plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    ax = plt.subplot()
    ax.set_xlim([-3, 5])
    x_ticks = np.arange(-3, 4)
    x_ticks_labels = [str(i) if i <= 3 else "" for i in x_ticks]
    plt.xticks(x_ticks, x_ticks_labels)
    ax.set_ylim([0, 40])
    ax.set_ylabel(r"$f_a(x)$")
    ax.set_xlabel("x")
    plt_save_figure(show_figure, save_path)


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """Generate graph for different goniometric functions

    Keyword arguments:
    show_figure -- bool deciding if the figure is shown on command line
    save_path -- string for saving the figure on the given path
    """
    t = np.linspace(start=0, stop=100, num=10000)
    # part for function 1
    f1 = 0.5 * np.cos(1 / 50 * np.pi * t)
    set_graph(r"$f_1(t)$")
    plt.plot(t, f1)
    plt_save_figure(show_figure, save_path)

    # part for function 2
    f2 = 0.25 * (np.sin(np.pi * t) + np.sin(3 / 2 * np.pi * t))
    set_graph(r"$f_2(t)$")
    plt.plot(t, f2)
    plt_save_figure(show_figure, save_path)

    # part for sum of function 1 and function 2
    f = f1 + f2
    ax = plt.subplot()
    set_graph(r"$f_1(t) + f_2(t)$")
    segment_start = 0
    segment_color = "green"
    above_treshold = f >= f1
    for i in range(1, len(t)):
        if above_treshold[i] != above_treshold[i - 1]:
            ax.plot(
                t[segment_start : i + 1], f[segment_start : i + 1], color=segment_color
            )
            segment_start = i
            segment_color = "green" if above_treshold[i] else "red"
    ax.plot(t[segment_start:], f[segment_start:], color=segment_color)
    plt_save_figure(show_figure, save_path)


def download_data() -> List[Dict[str, Any]]:
    """Scrape table from a website"""
    data = []
    req = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html")
    req.encoding = req.apparent_encoding

    soup = BeautifulSoup(req.text, "html.parser")
    tables = soup.find_all("table")
    table = tables[1]

    rows = table.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])

    output_data = []
    for row in data[1:]:
        output_data.append(
            {
                "position": row[0],
                "lat": to_float(row[1][:-1]),
                "long": to_float(row[3][:-1]),
                "height": to_float(row[5]),
            }
        )
    return output_data


def plt_save_figure(show_figure: bool, save_path: str):
    """Save figures based on the given info

    Keyword arguments:
    show_figure -- bool deciding if the figure is shown on command line
    save_path -- string for saving the figure on the given path
    """
    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
    plt.clf()


def set_graph(label: str):
    """Set up shared properties for the graphs

    Keyword arguments:
    label -- string that is written on the y axis of the graph
    """
    plt.ylim([-0.8, 0.8])
    plt.xlim([0, 100])
    ax = plt.subplot()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.set_ylabel(label)
    ax.set_xlabel(r"$t$")


def to_float(string: str) -> float:
    """Convert string to float regardless of commas"""
    return float(string.replace(",", "."))
