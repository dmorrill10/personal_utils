#!/usr/bin/env python

# Graphs input columns of numbers
# New columns of numbers can follow and they will all be aggregated but they
# must be
# separated by a commented line that is the graph's name
# ```
# # Graph Line 1
# ...
# # Graph Line 2
# ...
# ```
import re
import csv
import pandas as pd


def nonempty_lines(lines):
    for line in lines:
        line = line.strip()
        if line != '': yield line


class DataSet(object):
    @classmethod
    def read(self, lines, **csv_reader_kwargs):
        name = 'Data'
        data_lines = []
        pattern = re.compile('^\s*#\s*(.*)\s*$')
        if 'delimiter' not in csv_reader_kwargs:
            csv_reader_kwargs['delimiter'] = ' '
        if 'skipinitialspace' not in csv_reader_kwargs:
            csv_reader_kwargs['skipinitialspace'] = True
        for line in nonempty_lines(lines):
            result = pattern.match(line)
            if result is None:
                data_lines.append(line)
            else:
                if len(data_lines) > 0:
                    yield self(
                        name,
                        list(csv.reader(data_lines, **csv_reader_kwargs)))
                    data_lines = []
                name = result.group(1)
        if len(data_lines) > 0:
            yield self(name, list(csv.reader(data_lines, **csv_reader_kwargs)))

    def __init__(self, name, data):
        self.name = name
        self.data = pd.DataFrame(data).astype('float')


import itertools
import matplotlib


def color_table():
  # These are the "Tableau 20" colors as RGB.
  tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
               (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
               (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
               (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
               (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

  tableau40 = tableau20 + [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


  # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
  for i in range(len(tableau40)):
    r, g, b = tableau40[i]
    if i > 19:
      r = min([r + 16, 255])
      g = min([g + 16, 255])
      b = min([b + 16, 255])
    tableau40[i] = (r / 255., g / 255., b / 255.)
  return itertools.cycle(tableau40)

def marker_table():
  table = []
  for k,v in sorted(matplotlib.markers.MarkerStyle.markers.items(), key=lambda x: (str(x[0]), str(x[1]))):
    if (
      v != "nothing"
      and v != "pixel"
      and v != "vline"
      and v != 'hline'
#       v != 'star' and
      and v != 'tri_down'
      and v != 'tickleft'
      and v != 'tickup'
      and v != 'tri_up'
      and v != 'tickright'
#       v != 'x' and
      and v != 'tickdown'
      and v != 'tri_left'
      and v != 'tri_right'
      and v != 'plus'
    ):
      table.append(k)
  return itertools.cycle(table)


import matplotlib.pyplot as plt


def plot_estimation_error(file_name,
                          ylabel='',
                          xlabel='Epoch',
                          title='',
                          legend=None,
                          legend_loc='best',
                          ylim=None,
                          alpha=0.6,
                          marker_size=10,
                          mark_every=None,
                          line_width=2,
                          size=None):
    with open(file_name, 'r') as f:
        data_sets = list(DataSet.read(f))
    fig = plt.figure(figsize=size)
    i = 0
    colors = color_table()
    markers = marker_table()
    for s in data_sets:
        plt.plot(s.data[0],
                 s.data[1],
                 marker=next(markers),
                 markevery=mark_every,
                 color=next(colors),
                 alpha=alpha,
                 markersize=marker_size,
                 linewidth=line_width)
        if legend is None:
            print("{}: {}".format(i, s.name))
        i += 1
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if legend is None:
        plt.legend(list(range(len((data_sets)))), loc=legend_loc)
    else:
        plt.legend(legend(data_sets), loc=legend_loc)
    return fig


# if __name__ == '__main__':
#     import sys
#     import matplotlib.pyplot as plt
#
#     data_sets = list(DataSet.read(sys.stdin))
#     fig = plt.figure()
#     i = 0
#     for s in data_sets:
#         plt.plot(s.data[0], s.data[1])
#         print("{}: {}".format(i, s.name))
#         i += 1
#     plt.ylim([0.6926, 0.69277])
#     plt.legend(list(range(len(data_sets))))
#     plt.show()
