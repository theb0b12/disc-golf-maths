import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# i wanted to add this but it no work so idk
# df = pd.read_csv('disc_data.csv')


df = pd.DataFrame({
    'Disc Name': [
        'Judge', 'Marshal', 'Tailor', 'Atlas', 
        'Zombee', 'TL3', 'Orion LS', 
        'Bleak', 'Condor', 'Gote', 'Buzzz', "Wraith", "Eagle"
    ],
    'Speed': [2, 3, 4, 5, 6, 8, 9, 2, 3, 4, 5,11,7],
    'Glide': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4,5,4],
    'Turn': [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1,-1.1,-0.9],
    'Fade': [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,2.9, 2.9],
    'Total Distance': [229.9, 242.3, 259.3, 276.2, 293.2, 328.7, 345.7, 229.9, 245.4, 277.8, 276.2,402.8,311.7]
})


# extract features (X) and target (y)

X = torch.tensor(df[['Speed', 'Glide', 'Turn', 'Fade']].values, dtype=torch.float32)
y = torch.tensor(df['Total Distance'].values, dtype=torch.float32).unsqueeze(1)

# define the single layer NN
class DiscDistanceModel(nn.Module):
    def __init__(self):
        super(DiscDistanceModel, self).__init__()
        self.linear = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        return self.linear(x)

model = DiscDistanceModel()

# loss function
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005) # Lower learning rate for stability "IDK THIS ITS JUST WHAT ONLINE SAID"

epochs = 2000 # complete guess
for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# extract the multipliers and bias
weights = model.linear.weight.detach().numpy()[0]
bias = model.linear.bias.item()

print(f"Total Distance = ({weights[0]:.4f} * Speed) + ({weights[1]:.4f} * Glide) + ({weights[2]:.4f} * Turn) + ({weights[3]:.4f} * Fade) + {bias:.4f}")