import glob
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

TEAM_NAME_MAP = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
    'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Bobcats', 'CHO': 'Charlotte Hornets', 'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers', 'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NOH': 'New Orleans Hornets', 'NYK': 'New York Knicks',
    'NJN': 'New Jersey Nets',
    'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns', 'PHO': 'Phoenix Suns', 'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs', 'SEA': 'Seattle Supersonics', 'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
    # full name mappings
    'Detroit Pistons': 'Detroit Pistons',
    'Indiana Pacers': 'Indiana Pacers',
    'San Antonio Spurs': 'San Antonio Spurs',
    'New Jersey Nets': 'New Jersey Nets',
    'Dallas Mavericks': 'Dallas Mavericks'
}

#changed names:
# Charlotte Bobcats/Hornets
# New Orleans Pelicans/Hornets

# CLEANING PLAYER AND PLAYOFF DATA

def load_player_data(): #as supplied by raya
    # load all regular season player stats from Excel files
    player_files = glob.glob('Preprocessing/Preprocessed Data/Player Stats Regular and Playoff/*_filtered.xlsx')
    dfs = []
    
    for file in player_files:
        df = pd.read_excel(file)
        season = file.split('\\')[-1].split('_')[0]  # Extract season from filename
        df['Team'] = df['Team'].map(TEAM_NAME_MAP).fillna(df['Team'])
        df['Season'] = season

        #getting the columns we actually need
        df = df[['Player', 'Team', 'Season', "MP", "FG%", "3P%", "eFG%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]].copy()

        #filling in nan
        for col in ['FG%', '3P%', 'eFG%', 'FT%']:
            df[col] = df[col].fillna(0)
        #print(df.columns[df.isna().any()].tolist()) #['FG%', '3P%', 'eFG%', 'FT%']

        dfs.append(df)
        
    
    return pd.concat(dfs, ignore_index=True)

def load_playoff_data():

    team_files = glob.glob('Preprocessing/Preprocessed Data/Actual Playoff Team Stats/*__playoff_actual_team_stats.xlsx')
    dfs = []

    for file in team_files:
        if '~$' in file:
            continue
        df = pd.read_excel(file)
        season = file.split('\\')[-1].split('__')[0]
        df = df.rename(columns={'Tm': 'Team', 'Rk': 'Team Rank'})
        df['Season'] = season

        df = df[['Team Rank', 'Team', 'Season']].copy()

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

player_data = load_player_data()
playoff_data = load_playoff_data()

# MERGING PLAYER AND PLAYOFF DATA

def merge_data(player_data, playoff_data):

    player_df = player_data
    player_df['Team Rank'] = None
    dataframes = []

    for season in player_data.Season.unique():
        season_playoff_df = playoff_data[playoff_data['Season'] == season]
        season_player_df = player_df[player_df['Season'] == season]
        
        for row in season_player_df.itertuples():
            team = row.Team
            season_player_df.at[row.Index, 'Team Rank'] = season_playoff_df[season_playoff_df['Team'] == team].iloc[0]['Team Rank']
            
        dataframes.append(season_player_df)
    
    return pd.concat(dataframes)

    

merged_data = merge_data(player_data, playoff_data)
print(merged_data)

train_data = player_data
test_data = player_data
train = torch.tensor(train_data[["MP", "FG%", "3P%", "eFG%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]].values) #features
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
