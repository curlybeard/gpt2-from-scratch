import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig 

#------------------
#   PREPROCESSING
#------------------

ds = load_dataset("roneneldan/TinyStories")
# print(f"First train sample(only first 50 characters): {ds['train'][0]['text'][:50]} ")

train_subset = ds['train'].select(range(1000))
valid_subset = ds['validation'].select(range(100))

ds_small = {
    'train': train_subset,
    'validation': valid_subset
}

# print(f"Training stories: {len(ds_small['train'])}")
# print(f"Validation stories: {len(ds_small['validation'])}")

#byte pair encoding
encoder = tiktoken.get_encoding("gpt2")
#encode text to tokens
tokens = encoder.encode("Hello, I'm a language model,")
# print(f"Tokenzied input: {tokens}")

original_text = encoder.decode(tokens)
# print(f"De-tokenized tokens to original text: {original_text}")

class TinyStoriesDataset(Dataset):
    def __init__(self, split, encoder, context_length=128):
        self.split = split
        #convert text to numbers
        self.encoder = encoder
        #how many tokens the model sees at once
        self.context_length = context_length

        print(f"Tokenizing from {split}")
        #list to store tokens
        self.tokens = []

        #iterate through the dataset
        for i in range(len(ds_small[split])):
            #extract the text
            text = ds_small[split][i]['text']
            #convert to tokens
            tokens = self.encoder.encode(text)

            #convert list of tokens [10, 3, 255] to individual tokens
            self.tokens.extend(tokens)
            #add end of text integer 50256 at end of each story
            #acts as a contextual reset eg) lived happily ever after [EOT] Once upon a time...
            self.tokens.append(encoder.eot_token)
        
        #convert the token list into a tensor
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        # print(f"Total tokens: {len(self.tokens):,}")
        # print(f"Training chunks: {len(self.tokens)}")

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    #create next prediction token training examples for the model
    #model learns to predict next token given all previous tokens
    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        #eg. "The cat sat on"
        x = self.tokens[start:end]
        #eg. "cat sat on the"
        y = self.tokens[start+1:end+1]
        return x, y

#--------------------
#    TRAINING TIME
#--------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dataset = TinyStoriesDataset('train', encoder, context_length=128)

#randomize training order to prevent overfitting to story order
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

config = GPTConfig()
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()
for i in range(50):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        #clear old gradients
        optimizer.zero_grad()
        logits, loss = model(x_batch, y_batch)

        #back propgate to compute gradients for every parameter
        loss.backward()

        #gradient clipping for stable trainig, prevent gradient from exploding to huge value
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #update model weights using cthe computer gradient
        optimizer.step()

        print(f"step {i}, loss: {loss.item()}")

torch.save(model.state_dict(), "gpt2_tiny.pth")
print("Training complete. Model saved as gpt2_tiny.pth")