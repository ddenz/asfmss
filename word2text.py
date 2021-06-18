import docx
import os

pin = 'data/test.docx'
pout = os.path.splitext(pin)[0] + '.txt'

doc = docx.Document(pin)
text = '\n'.join([p.text for p in doc.paragraphs])

with open(pout, 'w') as fout:
    print(text, file=fout)
fout.close()
