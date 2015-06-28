#!/usr/bin/env ruby

# Given columns of numbers, computes the mean, stddev, and 95% CI
# New columns of numbers can follow and they will all be aggregated but they must be
# separated by a commented line of column names
# ```
# # d1 d2 d3
# ...
# # d1 d4 d5
# ...
# ```
NINETY_FIVE_PERCENT_CI_CONSTANT = 1.96
def compute_statistics
  ns = {}
  means = {}
  m2s = {}
  column_to_label = []
  STDIN.each do |line|
    line.strip!
    next if line.match(/^\s*$/)

    if line.match(/^\s*#/)
      line.split(/\s+/)[1..-1].each_with_index do |label, column|
        column_to_label[column] = label

        ns[label] ||= 0
        means[label] ||= 0
        m2s[label] ||= 0
      end

      next
    end

    line.split(/\s+/).each_with_index do |datum, column|
      next unless datum
      datum = datum.to_f

      ns[column_to_label[column]] ||= 0
      means[column_to_label[column]] ||= 0
      m2s[column_to_label[column]] ||= 0

      ns[column_to_label[column]] += 1
      delta = datum - means[column_to_label[column]]
      means[column_to_label[column]] += delta/ns[column_to_label[column]].to_f
      m2s[column_to_label[column]] += delta*(datum - means[column_to_label[column]])
    end
  end

  sample_variances = {}
  standard_errors = {}
  cis = {}

  m2s.each_pair do |label, m2|
    sample_variances[label] = m2 / (ns[label] - 1).to_f
    standard_errors[label] = (sample_variances[label] / ns[label].to_f)**(1/2.to_f)
    cis[label] = NINETY_FIVE_PERCENT_CI_CONSTANT * standard_errors[label]
  end

  {means: means, sample_variances: sample_variances, standard_errors: standard_errors, cis: cis, ns: ns}
end

def print_table(statistics)
  printf("# %25s%15s%15s%15s%15s%15s\n", '', "n", "mean", "s^2", "se", "95%_CI")

  statistics[:means].each_pair do |label, mean|
    printf("%27s%15d%15f%15f%15f%15f\n", label, statistics[:ns][label], mean, statistics[:sample_variances][label], statistics[:standard_errors][label], statistics[:cis][label])
  end
end

def print_ssv_with_errors(statistics, digits)
  statistics[:means].each_pair do |label, mean|
    printf("# %25s", label)
    puts
    printf '  '
    printf("%15s%10s\n", format("%0.#{digits}f", mean), format("%0.#{digits}f", statistics[:cis][label]))
  end
  puts
end

require 'optparse'


options = {}
OptionParser.new do |opts|
  opts.banner = "Usage: data | #{__FILE__} [options]"

  opts.on_tail("-h", "--help", "Show this message") do
    puts opts
    exit
  end

  options[:digits] = 5
  opts.on("-d", "--digits [NUMBER OF DIGITS]", "Number of digits after the decimal to print") do |d|
    options[:digits] = d.to_i || 5
  end

  options[:ssv] = false
  opts.on("-s", "--ssv", "Print as an SSV") do |s|
    options[:ssv] = s unless s.nil?
  end
end.parse!

statistics = compute_statistics

if options[:ssv]
  print_ssv_with_errors statistics, options[:digits]
else
  print_table statistics
end
