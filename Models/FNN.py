import glob
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 12 #MP, FG%, 3P%, eFG%, FT%, AST, STL, BLK, TOV, PF, PTS
hidden_size = 64 #64 to 128
batch = 16
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
        df = df[['Player', 'Team', 'Season', "MP", "FG%", "3P%", "eFG%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "G"]].copy()

        #filling in nan
        for col in ['FG%', '3P%', 'eFG%', 'FT%']:
            df[col] = df[col].fillna(0)
        #print(df.columns[df.isna().any()].tolist()) #['FG%', '3P%', 'eFG%', 'FT%']

        df = df.fillna(0)

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

    team_df = pd.concat(dfs, ignore_index=True)

    return team_df

player_data = load_player_data()
playoff_data = load_playoff_data()

# MERGING PLAYER AND PLAYOFF DATA AND PREPROCESSING

def merge_data(player_data, playoff_data):

    player_df = player_data

    #team_names = merged['Team'].unique().tolist()
    #le = LabelEncoder()
    #le.fit(team_names)
    #team_indices = dict(zip(le.classes_, range(len(le.classes_))))
    #we are really only concerned with the team and team rank, so we convert those to floats mapping to a unique int
    #merged['Team'] = le.transform(merged.Team.values)
    
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
    
    
    #assert not np.isnan(players_merged).any(), "NaN values in player arrays"
    #assert not np.isnan(playoff_merged).any(), "NaN values in team features"

    #individual stats, playoff team features for each player, target
    return player_data, playoff_merged

player_data, playoff_data = merge_data(player_data, playoff_data)

#making the model
num_layers = 4

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #input, 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size) #hidden
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout(p=0.3)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        for i in range(num_layers):
            out = self.l2(out)
            out = self.relu(out)
            out = self.drop(out) #lol
        out = self.l3(out)
        return out

#partitioning df by season
seasons = player_data.Season.unique()
print(seasons)
seasons_dict = {season : pd.DataFrame() for season in seasons}
for key in seasons_dict.keys():
    seasons_dict[key] = player_data[:][player_data.Season == key]

model = FNN(input_size, hidden_size, 1)
#always 12, always hidden_size, one single prediction

#loss and optimizer uses MSE and Adam
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# NEWLY ADDED OUTPUT SECTION
all_results = []

for season in seasons:
    season_dataset = seasons_dict[season]
    #processing season data
    X = season_dataset[["MP", "FG%", "3P%", "eFG%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]].values
    y = season_dataset[["Team Rank"]].values

    #standard scaler
    scaler = StandardScaler()
    #original_shape = train.shape
    #train_reshaped = train.reshape(-1, original_shape[-1])
    #train_reshaped = np.nan_to_num(train_reshaped, nan=0)
    #train = train_scaler.fit_transform(train).reshape(original_shape)
        
    #splitting into 80% training, 10% validation, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    #X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    #y_val = torch.tensor(y_val, dtype=torch.float32)
        
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    #num_teams = season_dataset['Team'].nunique()
    #print(f'{num_teams} teams in season dataset')

    #FNN actually training
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            #running_loss += loss.item() * X_batch.size(0)

            #if (i + 1) % 2 == 0:
            #    print(f'Season {season} Epoch {epoch+1}/{epochs}, batch {i}/{len(train_loader)}: loss: {loss.item():.4f}')

        #average_loss = loss / len(train_loader.dataset)
        #if (epoch+1) % 1 == 0:
        #    print(f"Season {season} -- Epoch {epoch+1}/{epochs} -- average loss: {average_loss:.4f}")
    
    #FNN evaluation and predicting using aggregated data
    model.eval()
    '''with torch.no_grad():
        #calculating loss
        test_preds = model(X_test)
        test_loss = criterion(test_preds, y_test)
    '''
    season_playoffs = playoff_data[playoff_data['Season'] == season]
    #    season_indices = season_playoffs.index.tolist()
    #    season_rankings = season_playoffs['Team Rank']
    columns = [
        'MP_sum', 'FG%_wt', '3P%_wt', 'eFG%_wt', 
            'FT%_wt', 'TRB_sum', 'AST_sum', 'STL_sum', 
            'BLK_sum', 'TOV_sum', 'PF_sum', 'PTS_sum'
            ]
    X_full = torch.tensor(season_playoffs[columns].values, dtype=torch.float32)
    full_predictions = model(X_full).detach().numpy().flatten()

    full_indices = season_playoffs.index.tolist()
    actual_ranks = np.argsort(np.argsort(season_playoffs['Team Rank'].values)) + 1
    predicted_ranks = np.argsort(np.argsort(full_predictions)) + 1
    '''
        season_stats = season_playoffs[columns]
        season_stats = torch.tensor(season_stats.values, dtype=torch.float32)
        
        #getting predictions
        predictions = model(season_stats).numpy().flatten()

        #results in a df
        results = pd.DataFrame({
            'Season': season,
            'Team': playoff_data['Team'].iloc[season_indices].tolist(),
            'Predicted Ranking': np.argsort(predictions) + 1,
            'Actual Ranking': season_rankings.values.tolist()
        })
        print(results)
    return test_loss.item()

overall_loss = []
for season in seasons:
    overall_loss.append(train(seasons_dict[season], season))
print(f'Overall average loss: {np.mean(overall_loss):.4f}')
'''
 # NEW: Output generation
    season_results = pd.DataFrame({
        'Season': season,
        'Team': season_playoffs['Team'].tolist(),
        'Actual_Rank': actual_ranks,
        'Predicted_Rank': predicted_ranks
    }).sort_values('Predicted_Rank')

    print(f"\n=== Season {season} Rankings ===")
    print(season_results[['Predicted_Rank', 'Team', 'Actual_Rank']].rename(columns={'Predicted_Rank': 'Rank'}).to_string(index=False))

    all_results.append(season_results)

if not all_results:
    raise ValueError("No valid seasons with sufficient data were processed")

import os
output_dir = "Results"
# NEW: Save all results to CSV
os.makedirs(output_dir, exist_ok=True)
final_results = pd.concat(all_results)
final_results.to_csv(os.path.join(output_dir, "FNN_complete_rankings.csv"), index=False)
print(f"\nAll seasons rankings saved to: {os.path.join(output_dir, 'FNN_complete_rankings.csv')}")
