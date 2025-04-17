import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import glob
from pathlib import Path

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = os.path.join(Path(__file__).parent.parent, "Preprocessing", "Preprocessed Data",)

# Hyper parameters
feature_cols = ["MP", "FG%", "3P%", "eFG%", "FT%", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
input_size = len(feature_cols)
hidden_size = 64
num_epochs = 50
batch_size = 16
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

def load_player_data(): #as supplied by raya
    player_files = glob.glob(os.path.join(PATH, "Player Stats Regular and Playoff", "*_filtered.xlsx"))
    dfs = []

    for file in player_files:
        df = pd.read_excel(file)
        season = file.split('\\')[-1].split('_')[0]  # Extract season from filename
        df['Team'] = df['Team'].map(TEAM_NAME_MAP).fillna(df['Team'])
        df['Season'] = season

        #getting the columns we actually need
        df = df[['Player', 'Team', 'Season', "MP", "FG%", "3P%", "eFG%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "G"]].copy()

        #filling in nan
        for col in ['FG%', '3P%', 'eFG%', 'FT%']:
            df[col] = df[col].fillna(0)
        #print(df.columns[df.isna().any()].tolist()) #['FG%', '3P%', 'eFG%', 'FT%']

        df = df.fillna(0)

        dfs.append(df)
        
    
    return pd.concat(dfs, ignore_index=True)

def load_playoff_data():

    team_files = glob.glob(os.path.join(PATH, "Actual Playoff Team Stats", "*__playoff_actual_team_stats.xlsx"))
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

    team_df = pd.concat(dfs, ignore_index=True)

    return team_df

player_data = load_player_data()
playoff_data = load_playoff_data()

# MERGING PLAYER AND PLAYOFF DATA AND PREPROCESSING

def merge_data(player_data, playoff_data):

    player_df = player_data
    
    playoff_data['Team Rank'] = playoff_data[['Team Rank']].astype(str).astype(float)

    ### aggregation by each team and season

    def weighted_avg(values, weights):
        mask = weights > 0
        if mask.any():
            return np.average(values[mask], weights=weights[mask])
        return 0

    aggregated_playoff = player_data.groupby(['Team', 'Season']).agg({
        'MP': 'sum',
        'FG%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        '3P%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'eFG%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'FT%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'TRB': 'sum',
        'AST': 'sum',
        'STL': 'sum',
        'BLK': 'sum',
        'TOV': 'sum',
        'PF': 'sum',
        'PTS': 'sum',
        'G': ['mean', 'std']
    }).reset_index()

    agg_columns = [
        'Team', 'Season', 'MP_sum', 'FG%_wt', '3P%_wt', 'eFG%_wt', 'FT%_wt',
        'TRB_sum', 'AST_sum', 'STL_sum', 'BLK_sum', 'TOV_sum',
        'PF_sum', 'PTS_sum', 'G_mean', 'G_std'
    ]
    aggregated_playoff.columns = agg_columns
    aggregated_playoff = aggregated_playoff.fillna(0)
    playoff_merged = pd.merge(aggregated_playoff, playoff_data, on=['Team', 'Season'], how='inner')

    # player array to turn into tensor later
    player_data['Team Rank'] = np.nan
    for i, row in player_data.iterrows():
        get_team = playoff_merged[(row['Team'] == playoff_merged['Team']) & (row['Season'] == playoff_merged['Season'])]
        player_data.loc[i, 'Team Rank'] = get_team['Team Rank'].values
    
    return player_data, playoff_merged

player_data, playoff_data = merge_data(player_data, playoff_data)



class TeamSequenceDataset(Dataset):
    def __init__(self, player_df, team_df, feature_cols):
        self.sequences = []
        self.targets = []
        self.meta = []

        grouped = player_df.groupby(['Team', 'Season'])
        for (team, season), group in grouped:
            group = group.sort_values(by='G', ascending=False)[:10]  # Top 10 players by games played
            if len(group) < 5:
                continue  # Skip teams with too few players

            features = group[feature_cols].values.astype(np.float32)
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            rank_row = team_df[(team_df['Team'] == team) & (team_df['Season'] == season)]
            if not rank_row.empty:
                self.sequences.append(torch.tensor(features))
                self.targets.append(rank_row['Team Rank'].values[0])
                self.meta.append((season, team))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.float32), self.meta[idx]

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out.squeeze()



# Dataset & Dataloader
def collate_fn(batch):
    sequences, targets, metas = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    targets = torch.stack(targets)
    return padded_sequences, targets, metas

dataset = TeamSequenceDataset(player_data, playoff_data, feature_cols)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model setup
model = RNNModel(input_size, hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for sequences, targets, _ in dataloader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# Evaluation and Ranking
model.eval()
results = []
with torch.no_grad():
    for i in range(len(dataset)):
        seq, target, (season, team) = dataset[i]
        seq = seq.unsqueeze(0).to(device)
        pred = model(seq).item()
        actual = target.item()
        results.append((season, team, actual, pred))

# Convert to DataFrame and compute rankings
results_df = pd.DataFrame(results, columns=["Season", "Team", "Actual_Rank", "Predicted_Score"])
results_df["Predicted_Rank"] = results_df.groupby("Season")['Predicted_Score'].rank(ascending=True).astype(int)
results_df["Actual_Rank"] = results_df["Actual_Rank"].astype(int)
results_df = results_df.sort_values(by=["Season", "Predicted_Rank"])

# Save to CSV
output_path = os.path.join(Path(__file__).parent.parent, "Results/Test/RNN_complete_rankings.csv")
results_df[['Season', 'Team', 'Actual_Rank', 'Predicted_Rank']].to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
