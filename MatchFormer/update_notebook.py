import nbformat as nbf

nb_file = 'Failure_Mode_Analysis.ipynb'
with open(nb_file, 'r') as f:
    nb = nbf.read(f, as_version=4)

# We just want to make sure the inline matplotlib instruction is at the top
# and that `plt.show()` is cleanly executed.

# The current notebook structure:
# Cell 0: Markdown
# Cell 1: Imports
# Cell 2: Markdown
# Cell 3: Model setup
# Cell 4: Markdown
# Cell 5: analyze_pair() definition
# Cell 6: Markdown
# Cell 7: Execution 1
# Cell 8: Markdown
# Cell 9: Execution 2

# Add `%matplotlib inline` to the imports cell
import_code = nb.cells[1].source
if '%matplotlib inline' not in import_code:
    nb.cells[1].source = '%matplotlib inline\n' + import_code

# Make sure plt.show() works smoothly and figure sizing is good
func_code = nb.cells[5].source
if 'plt.show()' not in func_code:
    nb.cells[5].source = func_code + '\n        plt.show()\n'

# Add an explicit output execution cell to run the whole thing now if they want it
# Actually, let's just use jupyter nbconvert to *execute* the notebook so it saves the outputs inside the file itself!

with open(nb_file, 'w') as f:
    nbf.write(nb, f)

print("Notebook updated for inline plotting.")
