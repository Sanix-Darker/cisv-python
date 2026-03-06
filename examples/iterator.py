import cisv

for row in cisv.open_iterator('examples/sample.csv'):
    print(row)
    if row and row[0] == '2':
        break
