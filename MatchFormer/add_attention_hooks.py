import nbformat as nbf

nb_path = '/Users/siddharthraj/classes/cv/final-proj/MatchFormer/tum_random_match.ipynb'
nb = nbf.read(nb_path, as_version=4)

code = """\
# Extracting Exact Cross-Attention Matrices
# MatchFormer uses linear attention: Q, K, V
# To get the explicit N x N attention matrix (which is O(N^2) and normally avoided),
# we can compute it manually for the cross-attention layers.

import torch.nn.functional as F

cross_attn_matrices = {}

def get_cross_attn(name, layer):
    def hook(model, input, output):
        # We need to intercept Q and K from the Attention module
        # Since we can't easily hook local variables, we'll recompute the attention matrix
        # using the module's own weights and the input it received.
        
        x = input[0]
        # Same logic as Attention.forward
        B, N, C = x.shape
        MiniB = B // 2
        
        query = layer.q(x).reshape(B, N, layer.num_heads, C // layer.num_heads).permute(0, 1, 2, 3)
        kv = layer.kv(x).reshape(B, -1, 2, layer.num_heads, C // layer.num_heads).permute(2, 0, 1, 3, 4)
        
        if layer.cross == True:
            k1, k2 = kv[0].split(MiniB)
            # Cross attention: key/value come from the other image
            key = torch.cat([k2, k1], dim=0)
        else:
            return # We only want cross-attention
            
        Q = layer.feature_map(query)
        K = layer.feature_map(key)
        
        # Q: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.permute(0, 2, 1, 3) 
        # K: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        K = K.permute(0, 2, 1, 3)

        # Compute explicit attention matrix: A = Q @ K^T
        # K.transpose: (B, num_heads, head_dim, N)
        # attn: (B, num_heads, N_query, N_key)
        attn = torch.matmul(Q, K.transpose(-2, -1))
        
        # Apply normalization (Z in the original code)
        Z = 1 / (torch.einsum("bnld,bnd->bnl", Q, K.sum(dim=2)) + layer.eps)
        # Z: (B, num_heads, N_query). Expand to divide each row.
        attn = attn * Z.unsqueeze(-1)
        
        cross_attn_matrices[name] = attn.detach().cpu()
        
    return hook

# Register hooks on the Attention module within each cross-attention Block
hook_handles_attn = []
# Stage 1 has cross=True on the 3rd block (index 2)
hook_handles_attn.append(model.matcher.backbone.AttentionBlock1.block[2].attn.register_forward_hook(
    get_cross_attn('stage1_cross', model.matcher.backbone.AttentionBlock1.block[2].attn)))

# Stage 2 has cross=True on the 3rd block (index 2)
hook_handles_attn.append(model.matcher.backbone.AttentionBlock2.block[2].attn.register_forward_hook(
    get_cross_attn('stage2_cross', model.matcher.backbone.AttentionBlock2.block[2].attn)))

# Stage 3 has cross=True on 2nd and 3rd blocks (index 1, 2)
hook_handles_attn.append(model.matcher.backbone.AttentionBlock3.block[1].attn.register_forward_hook(
    get_cross_attn('stage3_cross_1', model.matcher.backbone.AttentionBlock3.block[1].attn)))
hook_handles_attn.append(model.matcher.backbone.AttentionBlock3.block[2].attn.register_forward_hook(
    get_cross_attn('stage3_cross_2', model.matcher.backbone.AttentionBlock3.block[2].attn)))

# Stage 4 has cross=True on 2nd and 3rd blocks (index 1, 2)
hook_handles_attn.append(model.matcher.backbone.AttentionBlock4.block[1].attn.register_forward_hook(
    get_cross_attn('stage4_cross_1', model.matcher.backbone.AttentionBlock4.block[1].attn)))
hook_handles_attn.append(model.matcher.backbone.AttentionBlock4.block[2].attn.register_forward_hook(
    get_cross_attn('stage4_cross_2', model.matcher.backbone.AttentionBlock4.block[2].attn)))

print("Registered cross-attention specific hooks.")

with torch.no_grad():
    model.matcher(input_data)

for k, v in cross_attn_matrices.items():
    print(f"{k} attention matrix shape: {v.shape}")

for handle in hook_handles_attn:
    handle.remove()
"""

new_md = nbf.v4.new_markdown_cell("## Extract Exact Cross-Attention Matrices\nUsing a custom hook to recompute the exact $N \\times N$ attention maps from $Q$ and $K$.")
new_cell = nbf.v4.new_code_cell(code)
nb.cells.extend([new_md, new_cell])

with open(nb_path, 'w') as f:
    nbf.write(nb, f)

print("Injected exact cross-attention block into notebook.")
