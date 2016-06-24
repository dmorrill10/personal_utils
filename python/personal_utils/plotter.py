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


def to_hex(number):
    hex_alpha = list(range(10)) + ['A', 'B', 'C', 'D', 'E', 'F']
    return hex_alpha[number % len(hex_alpha)]


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
