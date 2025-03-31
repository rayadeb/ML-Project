import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============ CONFIGURATION ============
TEAM_NAME_MAP = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
    'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers', 'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns', 'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
    # full name mappings
    'Detroit Pistons': 'Detroit Pistons',
    'Indiana Pacers': 'Indiana Pacers',
    'San Antonio Spurs': 'San Antonio Spurs',
    'New Jersey Nets': 'New Jersey Nets',
    'Dallas Mavericks': 'Dallas Mavericks'
}

# ============ DATA LOADING ============
def load_player_stats(base_path):
    player_files = glob.glob(os.path.join(base_path, "Player Stats Regular and Playoff", "*_filtered.xlsx"))
    dfs = []
    
    for file in player_files:
        if '~$' in file:
            continue
            
        season = os.path.basename(file).split('_')[0]
        df = pd.read_excel(file)
        df['Team'] = df['Team'].map(TEAM_NAME_MAP).fillna(df['Team'])
        df['Season'] = season
        
        # select and clean relevant columns
        player_stats = df[['Player', 'Team', 'Season', 'G', 'MP', 'FG%', '3P%', 'eFG%', 'FT%',
                          'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].copy()
        
        # handle missing values
        for col in ['FG%', '3P%', 'eFG%', 'FT%']:
            player_stats[col] = player_stats[col].fillna(0)  # Assume 0% if no attempts
            
        # fill other missing values with 0 
        player_stats = player_stats.fillna(0)
        
        dfs.append(player_stats)
    
    return pd.concat(dfs, ignore_index=True)

def load_playoff_stats(base_path):
    playoff_files = glob.glob(os.path.join(base_path, "Actual Playoff Team Stats", "*__playoff_actual_team_stats.xlsx"))
    dfs = []
    
    for file in playoff_files:
        season = os.path.basename(file).split('__')[0]
        df = pd.read_excel(file)
        clean_df = df.rename(columns={'Tm': 'Team', 'Rk': 'Playoff_Rank'})[['Team', 'Playoff_Rank']].copy()
        clean_df['Team'] = clean_df['Team'].map(TEAM_NAME_MAP).fillna(clean_df['Team'])
        clean_df['Season'] = season
        dfs.append(clean_df)
    
    playoff_df = pd.concat(dfs, ignore_index=True)
    
    # validate playoff rankings
    for season in playoff_df['Season'].unique():
        season_ranks = playoff_df[playoff_df['Season'] == season]['Playoff_Rank']
        assert season_ranks.min() >= 1, f"Invalid rank <1 in {season}"
        assert len(season_ranks.unique()) == len(season_ranks), f"Duplicate ranks in {season}"
    
    return playoff_df

# ============ PREPROCESSING ============
def preprocess_data(player_df, playoff_df):
    # team aggregation with null handling
    def weighted_avg(values, weights):
        mask = weights > 0
        if mask.any():
            return np.average(values[mask], weights=weights[mask])
        return 0  # return 0 if no valid weights
    
    team_agg = player_df.groupby(['Team', 'Season']).agg({
        'MP': 'sum',
        'FG%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        '3P%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'eFG%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'FT%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'TRB': 'sum',
        'AST': ['sum', lambda x: x.nlargest(3).mean() if len(x) >= 3 else 0],
        'STL': 'sum',
        'BLK': 'sum',
        'TOV': 'sum',
        'PF': 'sum',
        'PTS': ['sum', lambda x: x.nlargest(3).mean() if len(x) >= 3 else 0],
        'G': ['mean', 'std']
    }).reset_index()
    
    # flatten multi-index columns
    team_agg.columns = [
        'Team', 'Season', 'MP_sum', 'FG%_wt', '3P%_wt', 'eFG%_wt', 'FT%_wt',
        'TRB_sum', 'AST_sum', 'AST_top3', 'STL_sum', 'BLK_sum', 'TOV_sum',
        'PF_sum', 'PTS_sum', 'PTS_top3', 'G_mean', 'G_std'
    ]
    
    # fill any remaining null values
    team_agg = team_agg.fillna(0)
    
    # merge with playoff data
    merged = pd.merge(team_agg, playoff_df, on=['Team', 'Season'], how='inner')
    
    # prepare player arrays
    player_cols = ['MP', 'FG%', '3P%', 'eFG%', 'FT%', 'TRB', 'AST', 
                   'STL', 'BLK', 'TOV', 'PF', 'PTS']
    player_arrays = []
    for _, row in merged.iterrows():
        team_players = player_df[(player_df['Team'] == row['Team']) & 
                               (player_df['Season'] == row['Season'])]
        arr = team_players[player_cols].values
        player_arrays.append(arr)
    
    # pad player arrays to uniform size
    max_players = max(arr.shape[0] for arr in player_arrays)
    player_arrays = np.stack([
        np.pad(arr, ((0, max_players - arr.shape[0]), (0, 0)), 
        mode='constant', constant_values=0)
        for arr in player_arrays
    ])
    
    # scale player stats with null protection
    player_scaler = StandardScaler()
    original_shape = player_arrays.shape
    player_arrays_reshaped = player_arrays.reshape(-1, original_shape[-1])
    player_arrays_reshaped = np.nan_to_num(player_arrays_reshaped, nan=0)
    player_arrays = player_scaler.fit_transform(player_arrays_reshaped).reshape(original_shape)
    
    # prepare team features
    team_features = merged.drop(['Team', 'Season', 'Playoff_Rank'], axis=1).values
    team_scaler = StandardScaler()
    team_features = np.nan_to_num(team_features, nan=0)
    team_features = team_scaler.fit_transform(team_features)
    
    # final validation
    assert not np.isnan(player_arrays).any(), "NaN values in player arrays"
    assert not np.isnan(team_features).any(), "NaN values in team features"
    assert not np.isnan(merged['Playoff_Rank'].values).any(), "NaN values in targets"
    
    return player_arrays, team_features, merged['Playoff_Rank'].values, merged
# ============ NEURAL NETWORK ============
class NBADataset(Dataset):
    def __init__(self, player_arrays, team_features, targets, original_indices=None):
        self.player_data = torch.FloatTensor(player_arrays)
        self.team_data = torch.FloatTensor(team_features)
        self.targets = torch.FloatTensor(targets)
        self.original_indices = original_indices if original_indices is not None else np.arange(len(targets))
    
    def __len__(self): 
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'player_stats': self.player_data[idx],
            'team_features': self.team_data[idx],
            'target': self.targets[idx],
            'original_idx': self.original_indices[idx]
        }

class HybridNBAModel(nn.Module):
    def __init__(self, n_players, n_player_features=12, n_team_features=16):
        super().__init__()
        # player pathway (1D CNN)
        self.player_net = nn.Sequential(
            nn.Conv1d(n_player_features, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Reduces player dimension to 1
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.Dropout(0.3)
        )
        # team pathway (fully connected)
        self.team_net = nn.Sequential(
            nn.Linear(n_team_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # combined network
        self.combined = nn.Sequential(
            nn.Linear(256, 128),  # 128 from player + 128 from team
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Final prediction
        )
    
    def forward(self, player_stats, team_features):
        # player stats: [batch, players, features] -> [batch, features, players]
        player_stats = player_stats.permute(0, 2, 1)
        player_out = self.player_net(player_stats)
        team_out = self.team_net(team_features)
        return self.combined(torch.cat([player_out, team_out], dim=1)).squeeze()

# ============ TRAINING ============

# ============ CUSTOM SUBSET CLASS ============
class NBASubset(torch.utils.data.Subset):
    def __getitem__(self, idx):
        original_item = self.dataset[self.indices[idx]]
        return {
            'player_stats': original_item['player_stats'],
            'team_features': original_item['team_features'],
            'target': original_item['target'],
            'original_idx': self.indices[idx]  # Preserve original index
        }

# ============ EVALUATION FUNCTION ============
def train_and_evaluate(base_path, output_dir="Results"):
    # load data
    player_df = load_player_stats(base_path)
    playoff_df = load_playoff_stats(base_path)
    
    # preprocess
    X_player, X_team, y, meta = preprocess_data(player_df, playoff_df)
    
    # create full dataset with original indices
    full_dataset = NBADataset(X_player, X_team, y)
    
    # group by season for proper evaluation
    seasons = sorted(meta['Season'].unique())
    all_results = []
    
    for season in seasons:
        print(f"\n=== Processing season {season} ===")
        season_mask = meta['Season'] == season
        season_indices = np.where(season_mask)[0]
        '''
        # skip seasons with too few teams
        if len(season_indices) < 4:
            print(f"Skipping season {season} - only {len(season_indices)} teams available")
            continue
        '''

        # create season dataset with correct original indices
        season_dataset = NBADataset(
            X_player[season_indices],
            X_team[season_indices],
            y[season_indices],
            original_indices=season_indices
        )
        
        # split dataset
        train_size = int(0.8 * len(season_dataset))
        train_set, test_set = torch.utils.data.random_split(
            season_dataset,
            [train_size, len(season_dataset) - train_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # create data loaders
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
        
        # initialize model
        model = HybridNBAModel(n_players=X_player.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # training
        model.train()
        for epoch in range(50):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['player_stats'], batch['team_features'])
                loss = criterion(outputs, batch['target'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        # get complete season predictions
        full_player = torch.stack([season_dataset[i]['player_stats'] for i in range(len(season_dataset))])
        full_team = torch.stack([season_dataset[i]['team_features'] for i in range(len(season_dataset))])
        full_indices = [season_dataset[i]['original_idx'] for i in range(len(season_dataset))]
        
        model.eval()
        with torch.no_grad():
            full_predictions = model(full_player, full_team).numpy()
        
        # create results dataframe
        team_names = meta['Team'].iloc[full_indices].tolist()
        actual_ranks = np.argsort(np.argsort(y[season_indices])) + 1
        predicted_ranks = np.argsort(np.argsort(full_predictions)) + 1
        
        season_results = pd.DataFrame({
            'Season': season,
            'Team': team_names,
            'Actual_Rank': actual_ranks,
            'Predicted_Rank': predicted_ranks
        }).sort_values('Predicted_Rank')
        
        print(f"\n=== Season {season} Rankings ===")
        print(season_results[['Predicted_Rank', 'Team', 'Actual_Rank']]
              .rename(columns={'Predicted_Rank': 'Rank'})
              .to_string(index=False))
        
        all_results.append(season_results)
    
    if not all_results:
        raise ValueError("No valid seasons with sufficient data were processed")

    os.makedirs(output_dir, exist_ok=True)

    final_results=pd.concat(all_results)
    output_path = os.path.join(output_dir, "HYBRID_complete_rankings.csv")
    final_results.to_csv(output_path, 
                       columns=['Season', 'Predicted_Rank', 'Team', 'Actual_Rank'],
                       index=False)
    print(f"\nAll seasons rankings saved to: {output_path}")
    
    return final_results
    
if __name__ == "__main__":
    results = train_and_evaluate("Preprocessing/Preprocessed Data")
    print(results.head(20))