import cisv

rows = cisv.parse_file('examples/sample.csv', delimiter=',', trim=True)
print(rows)
