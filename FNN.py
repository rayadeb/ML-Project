import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 12 #MP, FG%, 3P%, eFG%, FT%, AST, STL, BLK, TOV, PF, PTS
hidden_size = 64 #64 to 128
num_classes = 0 #will be number of teams
epochs = 2 #may set to higher value
batch_size = 100
learning_rate = 0.001

#parsing data first using pandas

excel_file = r"2023-24_filtered.xlsx"

train_data = pd.read_excel(excel_file, sheet_name="Regular")
train = torch.tensor(train_data[["MP", "EFG%", ""]].values)
test_data = pd.read_excel(excel_file, sheet_name="Regular")
test = torch.tensor(test_data[["PTS", "TRB", "AST"]].values)

#splitting into 80% training, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(train, test, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train, test):
        self.train = train
        self.test = test
    
    def __len__(self): #to enable len method
        return len(self.train)

train_ds = Dataset(X_train, y_train)
test_ds = Dataset(X_test, y_test)
valid_ds = Dataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=batch_size
)
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_ds,
    batch_size = batch_size
)

examples = iter(X_train)
samples = next(examples)

#making the model
num_layers = 2

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #input
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size) #hidden
        self.l3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        for i in range(num_layers):
            out = self.l2(out)
            out = self.relu(out)
        out = self.l3(out)

model = FNN(input_size, hidden_size, num_classes)

#loss and optimizer

#uses MSE and Adam
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#FNN actually training
for epoch in range(epochs):
    pass