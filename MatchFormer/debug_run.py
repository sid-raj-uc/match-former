import json

with open('WxBS_Visual_Analysis.ipynb', 'r') as f:
    nb = json.load(f)

with open('debug_script.py', 'w') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                f.write(line)
            f.write('\n\n')
