import torch
import torch.nn.functional as F
import tiktoken
from model import GPT, GPTConfig 

#-----------------------
#     GENERATING TEXT
#-----------------------

#initial setup with 5 sequences and max length of 30
num_return_sequences = 5
max_length = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load("gpt2_tiny.pth"))
model.to(device)
#disables dropout and batch norm updates
model.eval()

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
#add batch dimension [1,8] and duplicate to create 5 identical starting prompts [5, 8]
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

#keep generating tokens until 30
while x.size(1) < max_length:
    #save memory, no computation graph
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        #samples index from top 50 probability distribution
        ix = torch.multinomial(topk_probs, 1)
        #converts to vocab token ID
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

#convert each generated sequence back to text and display
for i in range(num_return_sequences):
    #slice to max length to ensure consistent output length
    tokens_list = x[i, :max_length].tolist()
    decoded = enc.decode(tokens_list)
    print(">", decoded)