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

def parse_table
  line = STDIN.gets.strip
  while line.match(/^\s*$/)
    line = STDIN.gets.strip
  end
  column_labels = line.split(/\s+/) + ['avg']
  $element_width = column_labels.inject(0) do |max_len, label|
    [label.length, max_len].max
  end + 4
  column_labels[0..-2].each do |label|
    printf("%#{$element_width}s", format("%s", label))
  end
  printf("%#{$element_width}s", format("%s", column_labels[-1]))
  puts ""
  column_labels.each do
    printf(" %#{$element_width-1}s", '-'*($element_width-1))
  end
  puts ""
  col_avgs = (column_labels.length - 1).times.map { 0.0 }
  row_avgs = []
  num_rows = 0
  STDIN.each do |line|
    line.strip!
    row_data = line.split(/\s+/)
    num_rows += 1
    row_avg = row_data.inject(0.0) do |avg, v|
      avg + v.to_f / (col_avgs.length - 1.0)
    end
    (row_data[1..-1] + [row_avg]).each_with_index do |value, i|
      col_avgs[i] += (value.to_f - col_avgs[i]) / num_rows.to_f
    end
    yield row_data[0], row_data[1..-1], row_avg
  end
  yield 'avg', col_avgs[0..-2], col_avgs[-1]
  column_labels.length
end

def print_table_row(row_label, means, plus_minus, num_digits_after_decimal)
  printf("%#{$element_width}s", format("`%s`", row_label))
  means.each_with_index do |mean, index|
    next unless index % 10 == 0

    printf("%#{$element_width}s", format("$%.#{num_digits_after_decimal}f \\pm %.#{num_digits_after_decimal}f$", mean, plus_minus[index]))
  end
  puts ""
end

def float_to_tex(value, num_digits_after_decimal, scale)
  v = value.to_f * scale
  v = 0 if v.abs < 1e-15
  format("$%.#{num_digits_after_decimal}f$", v)
end

def print_table_row_no_pm(row_label, values, avg, num_digits_after_decimal, scale)
  printf("%#{$element_width}s", format("%s", row_label))
  values.each_with_index do |value, index|
    printf(
      "%#{$element_width}s",
      float_to_tex(value, num_digits_after_decimal, scale))
  end
  printf(
    "%#{$element_width}s",
    float_to_tex(avg, num_digits_after_decimal, scale))
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
    options[:digits] = d.to_i || options[:digits]
  end

  options[:from_ssv_table] = false
  opts.on("--from-ssv-table", "Convert an SSV table to a pandoc table.") do |arg|
    options[:from_ssv_table] = arg unless arg.nil?
  end

  options[:scale] = 1.0
  opts.on("--scale [SCALE]", "A scaling factor with which to adjust all values in the table.") do |arg|
    options[:scale] = arg.to_f unless arg.nil?
  end
end.parse!

if options[:from_ssv_table]
  num_columns = parse_table do |row_label, values, avg|
    print_table_row_no_pm(row_label, values, avg, options[:digits], options[:scale])
  end
  (num_columns - 1).times.each do |index|
    # next unless index % 10 == 0
    printf(" %#{$element_width-1}s", '-'*($element_width-1))
  end
  printf(" %#{$element_width - 1}s", '-'*($element_width - 1))
  puts ""
else
  num_columns = parse_mean_data do |row_label, means, plus_minus|
    print_table_row(row_label, means, plus_minus, options[:digits])
  end
  num_columns.times.each do |index|
    next unless index % 10 == 0
    printf(" %#{$element_width-1}s", '-'*($element_width-1))
  end
  printf(" %#{$element_width-1}s", '-'*($element_width-1))
  puts ""
end
