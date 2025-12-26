import torch

try:
    import shap
except ImportError:
    !pip install shap >> shap_install.log
    import shap

from transformers import AutoTokenizer
checkpoint_str = "facebook/esm2_t30_150M_UR50D" #"facebook/esm2_t30_150M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_str)

def tokenizer_mine(seqs):
    seq_size = max([len(seq.split(" ")) for seq in seqs])
    max_len = seq_size+2
    
    x_feats = torch.zeros((len(seqs), max_len, 23)).to(device)
    x_feats[:, 0, 21] = 1
    masks = torch.zeros((len(seqs), max_len, 1)).to(device)
    for i, seq in enumerate(seqs):
        seq = [aa if aa in list(aa_alphabet) else "X" for aa in seq.split(" ")]
        x_feats[i, 1:seq_size+1, :21] = torch.FloatTensor(np.reshape(np.array(seq), (-1, 1)) == x_tokens)
        x_feats[i, seq_size+1, 22] = 1
        masks[i, 1:seq_size+1] = 1
    return x_feats, masks

def get_embeddings(model, tokenizer_mine, inp):
    x_feats, masks = tokenizer_mine(inp)
    results = model(x_feats, masks)
    return results

def factorize(number):
    candidate_factors, actual_factors = range(number), []
    for factor in candidate_factors:
        if number % factor == 0:
            actual_factors.append(factor)
          
    closest_1 = int(len(actual_factors) // 2)
    closest_0 = closest_1 - int(len(actual_factors) % 2 == 0)
    return actual_factors[closest_1], actual_factors[closest_0]

def visualize_shap(seq, model, tokenizer, full_class_str, figure_dims = ()):
    n_classes = len(full_class_str) 
    n_classes = n_classes if n_classes > 2 else 1
    if len(figure_dims) < 2:
        figure_dims = factorize(n_classes)
    n_rows, n_cols = figure_dims
  
    x_ =  [" ".join(list(seq))]
    shap_values = np.zeros((len(x_[0].split(" ")), len(x_[0].split(" ")), n_classes))
    for class_id in range(0, n_classes):
        explainer = shap.Explainer(lambda inp: get_embeddings(model, tokenizer, inp)[..., class_id], tokenizer)
        shap_values[..., class_id] = explainer(x_, fixed_context=1).values[0, 1:-1, 1:-1].T
    
    min_val, max_val = np.percentile(shap_values, [5, 95])
    max_val = np.max(np.abs([min_val, max_val]))
    min_val = -1*max_val
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize = (n_cols*8, n_rows*8))
    for class_id in range(n_classes):
      residual_shap_matrix = shap_values[..., class_id]
      ax[class_id//4, class_id%4].matshow(residual_shap_matrix, vmin = min_val, vmax = max_val, cmap = "coolwarm")
      ax[class_id//4, class_id%4].set_title(f"{full_class_str[class_id]} {min_val:.3f} to {max_val:.3f}")
