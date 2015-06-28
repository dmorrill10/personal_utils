#!/usr/bin/env ruby

def for_every_line_of_input
  STDIN.each do |line|
    line.strip!
    next if line.match(/^\s*$/)

    yield line
  end
end

# @yield A row of the eventual table
def for_every_row
  row_label = ""
  row_data = []
  for_every_line_of_input do |line|
    if line.match(/^\s*#\s*(.+)\s*$/)
      unless row_data.empty?
        yield row_label, row_data
      end
      row_label = $1
      row_data = []
      next
    end

    row_data << line.split(/\s+/)
  end
end

$element_width = 75

def print_column_headers(column_labels)
  ([''] + column_labels).each do |label|
    next unless label == '' || ((10*label.to_f) % 1 == 0)
    printf("%#{$element_width}s", format("`%s`", label))
  end
  puts ""
  (column_labels + ['']).each do |label|
    next unless label == '' || ((10*label.to_f) % 1 == 0)
    printf(" %#{$element_width-1}s", '-'*($element_width-1))
  end
  puts ""
end

# @return [Array] Array of row labels, means, and plus-minus
def parse_mean_data
  printed_column_headers = false
  num_columns = 0
  for_every_row do |row_label, row_data|
    unless printed_column_headers
      num_columns = row_data.length
      print_column_headers(row_data.map { |r| r[0].to_f })
      printed_column_headers = true
    end
    means = row_data.map { |r| r[1].to_f }
    plus_minus = row_data.map { |r| r[2].to_f }

    yield row_label, means, plus_minus
  end
  num_columns
end

def print_table_row(row_label, means, plus_minus, num_digits_after_decimal)
  printf("%#{$element_width}s", format("`%s`", row_label))
  means.each_with_index do |mean, index|
    next unless index % 10 == 0

    printf("%#{$element_width}s", format("$%.#{num_digits_after_decimal}f \\pm %.#{num_digits_after_decimal}f$", mean, plus_minus[index]))
  end
  puts ""
end

require 'optparse'

options = {}
OptionParser.new do |opts|
  opts.banner = "Usage: <ssv data> | #{__FILE__} [options]"

  opts.on("-h", "--help", "Show this message") do
    puts opts
    exit
  end

  options[:digits] = 2
  opts.on("-d", "--digits [NUMBER OF DIGITS]", "Number of digits after the decimal to print") do |d|
    options[:digits] = d.to_i || 2
  end
end.parse!

num_columns = parse_mean_data do |row_label, means, plus_minus|
  print_table_row(row_label, means, plus_minus, options[:digits])
end
num_columns.times.each do |index|
  next unless index % 10 == 0
  printf(" %#{$element_width-1}s", '-'*($element_width-1))
end
printf(" %#{$element_width-1}s", '-'*($element_width-1))
puts ""
