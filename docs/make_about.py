
with open('../README.rst','r') as f:
    text = f.read()

text = text.replace('epipack\n=======','About\n=====')

with open('./about.rst','w') as f:
    f.write(text)
