import torch
import nbformat as nbf

nb_path = '/Users/siddharthraj/classes/cv/final-proj/MatchFormer/tum_random_match.ipynb'
nb = nbf.read(nb_path, as_version=4)

code = """\
# Modified PyTorch forward hooks to extract attention probabilities
attn_weights = {}

def get_attn_hook(name):
    def hook(model, input, output):
        # We need to hook into the Attention module itself, or monkey-patch it
        pass
    return hook

"""
