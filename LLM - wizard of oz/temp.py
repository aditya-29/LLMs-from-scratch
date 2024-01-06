with open("wizard_of_oz.txt", "r", encoding = "utf-8") as f:
    text = f.read()

print("total length of text : ", len(text))
print(text[:200])

chars = sorted(list(set(text)))
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)
print()
print("total unique characters : ", vocab_size)

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : "".join([itos[i] for i in l])

e = encode("Hello World")
d = decode(e)

print(e)
print(d)


import torch
import torch.nn as nn
import torch.nn.functional as F

data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape)


# HYPER PARAMETER
device = "mps"
batch_size = 4
block_size = 8
learning_rate = 3e-4
max_iters = 10000
eval_iters = 500
dropout = 0.2
n_emb = 384
n_layer = 4
n_head = 4

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

x,y = get_batch("train")
print("input")
print(x)
print("targets")
print(y)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Decoder transformer architecture

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    

class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias = False)                            # (C, C // head_size)
        self.query = nn.Linear(n_emb, head_size, bias = False)                          # (C, C // head_size)
        self.value = nn.Linear(n_emb, head_size, bias = False)                          # (C, C // head_suze)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # get the 'key' representation
        k = self.key(x)             # (B, T, head_size)
        # get the 'query' representation
        q = self.query(x)           # (B, T, head_size)
        # get the 'value' representation
        v = self.value(x)           # (B, T, head_size)

        # mat mul of key and query 
        wei = q @ k.transpose(-2, -1)                           # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # mask fill the key-query pair
        wei = wei.masked_fill(self.tril == 0, float("-inf"))    # (B, T, T)
        # apply softmax
        wei = F.softmax(wei, dim=-1)                            # (B, T, T)

        # calculate the out matrix
        out = wei @ v               # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out

    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size=head_size) for _ in range(n_head)])
        self.droupout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_head * head_size, n_emb)

    def forward(self, x):
        # concatenate the output from all the heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # apply dropout
        out = self.droupout(out)
        # perform a linear transformation
        out = self.proj(out)

        return out


class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        # distribute the calculations evenly based on the head size
        head_size = n_emb // n_head   
        self.self_attn = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        # checkpoint for the residual connection
        y = self.self_attn(x)                                   # (B, T, C)
        # add the residual checkpoint to the current layer
        x = self.ln1(x + y)                                     # (B, T, C)
        # checkpoint for the residual connection
        y = self.ffwd(x)                                        # (B, T, C)
        # add the residual checkpoint to the current layer
        x = self.ln2(x + y)                                     # (B, T, C)

        return x




class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # converting each character into a series of embedding eg: 'a' can be represented by [0, 1, 2, ... n_emb]
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)       # (vocab_size, n_emb)   
        # convertng each position in the embedding table into eg: 0th element can be represented by [0, 0, 0, 0, 0 ... n_emb]
        self.position_embedding_table = nn.Embedding(block_size, n_emb)    # (vocab_size, vocab_size)
        # how many 'Decoder blocks' does this model contain ?
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        # final layer norm
        self.ln_f = nn.LayerNorm(n_emb)
        # final linear layer for the language model
        self.lm_head = nn.Linear(n_emb, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, indx, targets=None):                                  # (B, T), (B, T)
        logits = self.token_embedding_table(indx)                           # (B, T, n_emb)
        loss = None

        # indx and targets are (B, T) tensor of integers
        B, T = indx.shape
        # get the token emebddings
        tok_emb = self.token_embedding_table(indx)                          # (B, T, n_emb) -> this can also be represented by (B, T, C)
        # get the positional embeddings -> rows should have same dimention as 'T' each row is represented by 'n_emb' tensors
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))   # (T, C) -> this can also be represented by (B, T, C)
        # Add both positional and token embedding as 
        x = tok_emb + pos_emb                                               # (B, T, C)
        # pass the cocktail of pos and token embedding to the Transformer Blocks
        x = self.blocks(x)                                                  # (B, T, C)
        # apply layer norm on the final output from the Transformer Blocks
        x = self.ln_f(x)                                                    # (B, T, C)
        # calcualte the logits by applying a linear transformation
        logits = self.lm_head(x)                                            # from (B, T, C {n_emb}) to (B, T, vocab_size)

        if targets is None:
            return logits, loss
        else:
            B, T, C = logits.shape  
            logits = logits.view(B*T, C)                                        # Since cross entropy accepts the tensors in the shape (B, C, T) we are squeezing the dimentions
            targets = targets.view(B*T)                                          # squeezing the targets dimention into 1D
            
            loss = F.cross_entropy(logits, targets)

            return logits, loss
    
    def generate(self, indx, max_new_tokens):
        # index is a (B, T) tensor consisting of the inputs 
        for _ in range(max_new_tokens):
            # get the output from the model
            logits, loss = self(indx)
            # focus only on the last time step
            logits = logits[:, -1, :]                                           # (B, T)
            # apply the softmax on the logits to get the probabilities on the last dim
            probs = F.softmax(logits, dim=-1)                                   # (B, T)
            # choose a index based on the above calculated probability
            indx_nxt = torch.multinomial(probs, num_samples=1)                  # (B, 1)
            # append the new index to the current index array
            indx = torch.cat((indx, indx_nxt), dim=1)                             #(B, T+1)
        return indx
    
def count_params(model):
    s = 0
    for p in model.parameters():
        if p.requires_grad:
            s += p.numel()

    return s


model = GPTLanguageModel(vocab_size=vocab_size)
model = model.to(device)
print("Total parameters : ", count_params(model))

# Now traning the model to generate non random outputs
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
losses = []

for iter in range(max_iters):
    # sample a batch of training data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # track stats
    if iter % eval_iters == 0:
        _loss = estimate_loss()
        print(f"{iter} / {max_iters} loss = {_loss}")
print(losses[-1])

