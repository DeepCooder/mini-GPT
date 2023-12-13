import torch
import torch.nn as nn
from torch.nn import functional as F

# Define hyperparameters for the model
batch_size = 16 # Number of independent sequences processed in parallel
block_size = 32 # Maximum context length for predictions
max_iters = 5000 # Maximum number of iterations for training
eval_interval = 100 # Interval at which to evaluate the model
learning_rate = 1e-3 # Learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available, else use CPU
eval_iters = 200 # Number of iterations for evaluation
n_embd = 64 # Size of the embeddings
n_head = 4 # Number of attention heads
n_layer = 4 # Number of layers in the model
dropout = 0.0 # Dropout rate

# Set the seed for reproducibility
torch.manual_seed(1337)

# Load the text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars) # Size of the vocabulary
# Create a mapping from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) } # String to integer mapping
itos = { i:ch for i,ch in enumerate(chars) } # Integer to string mapping
encode = lambda s: [stoi[c] for c in s] # Function to encode a string into a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Function to decode a list of integers into a string

# Split the data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # Use 90% of the data for training
train_data = data[:n]
val_data = data[n:]

# Function to generate a batch of data
def get_batch(split):
    # Choose the correct dataset
    data = train_data if split == 'train' else val_data
    # Randomly select a starting point for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create the input and target sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Move the data to the correct device
    x, y = x.to(device), y.to(device)
    return x, y

# Function to estimate the loss of the model
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Set the model to evaluation mode
    # Calculate the loss for both the training and validation sets
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set the model back to training mode
    return out

# Define the self-attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Define the key, query and value linear transformations
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Create a lower triangular matrix for masking
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        # Apply the linear transformations
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute the attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # Apply the mask to the scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # Apply the softmax function to the scores
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # Apply dropout to the scores
        wei = self.dropout(wei)
        # Compute the weighted sum of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# Define the multi-head self-attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create the attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Define the output linear transformation
        self.proj = nn.Linear(n_embd, n_embd)
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs of the attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply the output linear transformation and dropout
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 
    A simple feed-forward neural network which consists of a linear layer followed by a non-linearity (ReLU).
    This is used as a computation layer in the transformer block.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Linear layer that expands the input dimension by 4
            nn.ReLU(),  # Non-linear activation function
            nn.Linear(4 * n_embd, n_embd),  # Linear layer that reduces the dimension back to the original
            nn.Dropout(dropout),  # Dropout layer for regularization
        )

    def forward(self, x):
        return self.net(x)  # Pass the input through the network

class Block(nn.Module):
    """ 
    Transformer block which consists of a self-attention mechanism (communication) followed by a feed-forward network (computation).
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # Size of each attention head
        self.sa = MultiHeadAttention(n_head, head_size)  # Multi-head self-attention mechanism
        self.ffwd = FeedFoward(n_embd)  # Feed-forward network
        self.ln1 = nn.LayerNorm(n_embd)  # Layer normalization before self-attention
        self.ln2 = nn.LayerNorm(n_embd)  # Layer normalization before feed-forward network

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Apply layer normalization, self-attention and add the residual connection
        x = x + self.ffwd(self.ln2(x))  # Apply layer normalization, feed-forward network and add the residual connection
        return x

class miniGPT(nn.Module):
    """
    A simplified version of the GPT model. It consists of an embedding layer, multiple transformer blocks, and a final linear layer.
    """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Embedding layer for tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Embedding layer for positions
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Final linear layer to predict the next token

    def forward(self, idx, targets=None):
        B, T = idx.shape  # Batch size and sequence length
        tok_emb = self.token_embedding_table(idx)  # Token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Position embeddings
        x = tok_emb + pos_emb  # Add token and position embeddings
        x = self.blocks(x)  # Pass through the transformer blocks
        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x)  # Get the logits for the next token prediction

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  # Compute the cross-entropy loss if targets are provided

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens given a context. The context is a sequence of token indices.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Crop the context to the last block_size tokens
            logits, loss = self(idx_cond)  # Get the predictions for the next token
            logits = logits[:, -1, :]  # Focus only on the last time step
            probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample the next token from the distribution
            idx = torch.cat((idx, idx_next), dim=1)  # Append the sampled token to the context
        return idx

model = miniGPT()  # Instantiate the model
m = model.to(device)  # Move the model to the device (GPU or CPU)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')  # Print the number of parameters in the model

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Create an optimizer

for iter in range(max_iters):  # Training loop
    if iter % eval_interval == 0 or iter == max_iters - 1:  # Evaluate the model periodically
        losses = estimate_loss()  # Estimate the loss on the training and validation sets
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')  # Sample a batch of training data
    logits, loss = model(xb, yb)  # Compute the logits and loss
    optimizer.zero_grad(set_to_none=True)  # Reset the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the model parameters

context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Create a context of zeros
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))  # Generate new tokens and print the decoded text
