from pathlib import Path
import xml.etree.ElementTree as ET
root = Path('/home/wwd/py_code/mhr/final/new_grn')
tree = ET.parse(root / 'reference/paper.xml')
for sec in tree.findall('.//sec'):
    title = sec.find('title')
    label = ''.join(title.itertext()) if title is not None else ''
    if any(x in label.lower() for x in ['coarse-grain', 'motif', 'integration of transcriptome', 'linear model-based', 'candidate regions', 'glycolysis']):
        print('\n', label)
        print(' '.join(''.join(p.itertext()) for p in sec.findall('p')))
for s in tree.findall('.//supplementary-material'):
    print('SUPPLEMENT', ET.tostring(s, encoding='unicode'))
