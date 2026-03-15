import nbformat as nbf
nb_file = 'Constrained_Inference_Analysis.ipynb'
with open(nb_file, 'r') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Replace the literal newline inside the f-string with a \n escape sequence
        # The syntax error "unterminated f-string literal" happens because notebooks 
        # sometimes wrap the string literally across two lines if not escaped properly.
        cell.source = cell.source.replace('axes[1].set_title(f"CONSTRAINED Stage 4 Attention\\nEAR: {ear_val:.4f}")', 'axes[1].set_title(f"CONSTRAINED Stage 4 Attention\\nEAR: {ear_val:.4f}")')
        # Wait, inside the f-string we just need to ensure the \n is there instead of actual newline.
        # Let's just do a manual string replace of the exact broken block if it exists
        if 'axes[1].set_title(f"CONSTRAINED Stage 4 Attention\nEAR' in cell.source:
             cell.source = cell.source.replace('axes[1].set_title(f"CONSTRAINED Stage 4 Attention\nEAR: {ear_val:.4f}")', 'axes[1].set_title(f"CONSTRAINED Stage 4 Attention\\nEAR: {ear_val:.4f}")')

with open(nb_file, 'w') as f:
    nbf.write(nb, f)
