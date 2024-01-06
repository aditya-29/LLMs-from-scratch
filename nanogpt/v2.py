import torch
import torch.nn as nn
import torch.nn.functional as F

# hyper parameters
batch_size = 64     # number of training examples in a batch
block_size = 256      # what is the maximum context length for prediction ?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "mps"
eval_iters = 200
n_emb = 384

n_head = 6
n_layer = 6
dropout = 0.2

if device == "mps":
    print("Using Metal GPU")
# -------------
    
torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from character taht occur in the text
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
encode = lambda s : [stoi[ch] for ch in s]
decode = lambda l : "".join([itos[i] for i in l])

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    '''
        One head of the self attention
    '''
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape     # here C == head_size

 
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # compute tyhe attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T)  -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v   # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    

class MultiHeadAttention(nn.Module):
    """
        multiple attention in parellel
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """
        a simple linear layer followed by a non-linearity
    """

    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        # self.sa_heads = MultiHeadAttention(4, n_emb//4)     # 4 heads of 8 dimensional self attention
        # self.ffwd = FeedForward(n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targer are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb   # (B, T, C)
        # x = self.sa_heads(x)     # apply one head of self attention. (B, T, C)
        # x = self.ffwd(x)         # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(tok_emb)              # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the last block size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]            # (B, C)
            # apply softmax to the last layer
            probs = torch.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append the sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = BigramLanguageModel(vocab_size)
print("Total params : ", count_params(model))
m = model.to(device)


# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a whle evaluate the loss on train and val sets
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter} : train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=  device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


