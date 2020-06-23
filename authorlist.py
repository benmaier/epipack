import csv

with open('authorlist.txt','r') as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
        data.append([row[1].lstrip().rstrip(), row[0], row[2].lstrip().rstrip()])

data = sorted(data, key=lambda x: x[1] + " " + x[0])
authors = [_[0] + " " + _[1] for _ in data]
standard = '"In alphabetical order: ' + ', '.join(authors) + '"'
readme = ",\n".join(['[{}]({})'.format(_[0] + " " + _[1],  _[2]) for _ in data])+"."

print("\n=========\n")
print(standard)
print("\n=========\n")
print(readme)
print()
