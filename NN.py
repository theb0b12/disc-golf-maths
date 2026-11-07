import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# i wanted to add this but it no work so idk
# df = pd.read_csv('disc_data.csv')


df = pd.DataFrame({
    'Disc Name': ['Beast', 'Eagle', 'Escape', 'Crave', 'Truth', 'Zone', 'Judge'],
    'Speed': [10, 7, 9, 6.5, 5, 4, 2],
    'Glide': [5, 4, 5, 5, 5, 4, 4],
    'Turn': [-2, -1, -1, -1, -1, 0, 0],
    'Fade': [2, 3, 2, 1, 1, 3, 1],
    'Total Distance': [390, 305, 370, 310, 295, 240, 215]
})


# extract features (X) and target (y)

X = torch.tensor(df[['Speed', 'Glide', 'Turn', 'Fade']].values, dtype=torch.float32)
y = torch.tensor(df['Total Distance'].values, dtype=torch.float32).unsqueeze(1)

# define the single layer NN
class DiscDistanceModel(nn.Module):
    def __init__(self):
        super(DiscDistanceModel, self).__init__()
        self.linear = nn.Linear(in_features=4, out_features=1, bias=False)

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

print(f"Total Distance = ({weights[0]:.4f} * Speed) + ({weights[1]:.4f} * Glide) + ({weights[2]:.4f} * Turn) + ({weights[3]:.4f} * Fade)")