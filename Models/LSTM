import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # For season-level split
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
    'Detroit Pistons': 'Detroit Pistons', 'Indiana Pacers': 'Indiana Pacers',
    'San Antonio Spurs': 'San Antonio Spurs', 'New Jersey Nets': 'New Jersey Nets',
    'Dallas Mavericks': 'Dallas Mavericks'
}
RANDOM_SEED = 42 # For reproducibility

# ============ DATA LOADING ============
def load_player_stats(base_path):
    """Loads and preprocesses player statistics from Excel files."""
    player_files = glob.glob(os.path.join(base_path, "Player Stats Regular and Playoff", "*_filtered.xlsx"))
    dfs = []
    for file_path in player_files:
        if '~$' in file_path: continue
        season = os.path.basename(file_path).split('_')[0]
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading player stats file {file_path}: {e}")
            continue
        df['Team'] = df['Team'].map(TEAM_NAME_MAP).fillna(df['Team'])
        df['Season'] = season
        player_stats_cols = ['Player', 'Team', 'Season', 'G', 'MP', 'FG%', '3P%', 'eFG%', 'FT%',
                             'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
        for col in player_stats_cols:
            if col not in df.columns: df[col] = 0
        player_stats = df[player_stats_cols].copy()
        for col in ['FG%', '3P%', 'eFG%', 'FT%']:
            player_stats[col] = player_stats[col].fillna(0)
        player_stats = player_stats.fillna(0)
        dfs.append(player_stats)
    if not dfs: raise ValueError(f"No player data loaded from {base_path}.")
    return pd.concat(dfs, ignore_index=True)

def load_playoff_stats(base_path):
    """Loads actual playoff team rankings."""
    playoff_files = glob.glob(os.path.join(base_path, "Actual Playoff Team Stats", "*__playoff_actual_team_stats.xlsx"))
    dfs = []
    for file_path in playoff_files:
        if '~$' in file_path: continue
        season = os.path.basename(file_path).split('__')[0]
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading playoff stats file {file_path}: {e}")
            continue
        # Ensure 'Tm' and 'Rk' columns exist
        if 'Tm' not in df.columns or 'Rk' not in df.columns:
            print(f"Skipping playoff file {file_path} due to missing 'Tm' or 'Rk' columns.")
            continue

        clean_df = df.rename(columns={'Tm': 'Team', 'Rk': 'Playoff_Rank'})[['Team', 'Playoff_Rank']].copy()
        clean_df['Team'] = clean_df['Team'].map(TEAM_NAME_MAP).fillna(clean_df['Team'])
        clean_df['Season'] = season
        dfs.append(clean_df)
    if not dfs: raise ValueError(f"No playoff data loaded from {base_path}.")
    playoff_df = pd.concat(dfs, ignore_index=True)
    # Validate playoff rankings
    for season in playoff_df['Season'].unique():
        season_ranks = playoff_df[playoff_df['Season'] == season]['Playoff_Rank']
        if not pd.api.types.is_numeric_dtype(season_ranks):
            print(f"Warning: Non-numeric Playoff_Rank found in season {season}. Attempting conversion.")
            season_ranks = pd.to_numeric(season_ranks, errors='coerce')
            playoff_df.loc[playoff_df['Season'] == season, 'Playoff_Rank'] = season_ranks
        
        season_ranks = season_ranks.dropna() # Remove NaNs after conversion for validation
        if not season_ranks.empty:
            assert season_ranks.min() >= 1, f"Invalid rank <1 in {season}"
            # Allow duplicate ranks if they exist in raw data, ranking logic will handle it.
            # assert len(season_ranks.unique()) == len(season_ranks), f"Duplicate ranks in {season}" 
    return playoff_df

# ============ PREPROCESSING ============
def preprocess_data(player_df, playoff_df):
    """
    Preprocesses player data, aggregates to team-level, and merges with playoff ranks.
    Returns player arrays, team features, target ranks, and a metadata DataFrame.
    """
    def weighted_avg(values, weights):
        mask = weights > 0
        if mask.any():
            return np.average(np.array(values)[mask], weights=np.array(weights)[mask])
        return 0
    
    team_agg = player_df.groupby(['Team', 'Season']).agg(
        MP_sum=('MP', 'sum'),
        FG_wt_avg=('FG%', lambda x: weighted_avg(x, player_df.loc[x.index, 'G'])),
        P3_wt_avg=('3P%', lambda x: weighted_avg(x, player_df.loc[x.index, 'G'])),
        eFG_wt_avg=('eFG%', lambda x: weighted_avg(x, player_df.loc[x.index, 'G'])),
        FT_wt_avg=('FT%', lambda x: weighted_avg(x, player_df.loc[x.index, 'G'])),
        TRB_sum=('TRB', 'sum'),
        AST_sum=('AST', 'sum'),
        AST_top3=('AST', lambda x: x.nlargest(3).mean() if len(x) >= 3 else (x.mean() if not x.empty else 0)),
        STL_sum=('STL', 'sum'),
        BLK_sum=('BLK', 'sum'),
        TOV_sum=('TOV', 'sum'),
        PF_sum=('PF', 'sum'),
        PTS_sum=('PTS', 'sum'),
        PTS_top3=('PTS', lambda x: x.nlargest(3).mean() if len(x) >= 3 else (x.mean() if not x.empty else 0)),
        G_mean=('G', 'mean'),
        G_std=('G', 'std')
    ).reset_index()
    
    team_agg.columns = [ # 18 columns
        'Team', 'Season', 'MP_sum', 'FG%_wt', '3P%_wt', 'eFG%_wt', 'FT%_wt',
        'TRB_sum', 'AST_sum', 'AST_top3_agg', 'STL_sum', 'BLK_sum', 'TOV_sum',
        'PF_sum', 'PTS_sum', 'PTS_top3_agg', 'G_mean', 'G_std'
    ]
    team_agg = team_agg.fillna(0)
    
    merged_df = pd.merge(team_agg, playoff_df, on=['Team', 'Season'], how='inner')
    merged_df['Playoff_Rank'] = pd.to_numeric(merged_df['Playoff_Rank'], errors='coerce').fillna(merged_df['Playoff_Rank'].max() + 1) # Handle any non-numeric ranks

    y_target = merged_df['Playoff_Rank'].values
    
    player_cols = ['MP', 'FG%', '3P%', 'eFG%', 'FT%', 'TRB', 'AST', 
                   'STL', 'BLK', 'TOV', 'PF', 'PTS'] # 12 features
    player_arrays_list = []
    for _, row in merged_df.iterrows():
        team_players = player_df[(player_df['Team'] == row['Team']) & (player_df['Season'] == row['Season'])]
        arr = team_players[player_cols].values
        player_arrays_list.append(arr)
    
    if not player_arrays_list: raise ValueError("No player arrays created.")
    max_players = max(arr.shape[0] for arr in player_arrays_list if arr.ndim > 0 and arr.shape[0] > 0)
    
    player_arrays_padded = []
    for arr in player_arrays_list:
        if arr.ndim == 0 or arr.shape[0] == 0:
             player_arrays_padded.append(np.zeros((max_players, len(player_cols))))
        else:
            player_arrays_padded.append(np.pad(arr, ((0, max_players - arr.shape[0]), (0, 0)), mode='constant'))
    X_player = np.stack(player_arrays_padded)

    player_scaler = StandardScaler()
    X_player_reshaped = X_player.reshape(-1, X_player.shape[-1])
    X_player_reshaped = np.nan_to_num(X_player_reshaped)
    X_player_scaled = player_scaler.fit_transform(X_player_reshaped)
    X_player = X_player_scaled.reshape(X_player.shape)
    
    team_feature_cols = [col for col in team_agg.columns if col not in ['Team', 'Season']] # 16 features
    X_team_df = merged_df[team_feature_cols] # Use merged_df to align with y_target
    X_team = X_team_df.values
    
    team_scaler = StandardScaler()
    X_team = np.nan_to_num(X_team)
    X_team = team_scaler.fit_transform(X_team)
    
    assert not np.isnan(X_player).any(), "NaNs in player arrays after scaling"
    assert not np.isnan(X_team).any(), "NaNs in team features after scaling"
    assert not np.isnan(y_target).any(), "NaNs in target playoff ranks"
    
    return X_player, X_team, y_target, merged_df # merged_df is our meta_df

# ============ NEURAL NETWORK ============
class NBADataset(Dataset):
    def __init__(self, player_arrays, team_features, targets, meta_df_indices):
        self.player_data = torch.FloatTensor(player_arrays)
        self.team_data = torch.FloatTensor(team_features)
        self.targets = torch.FloatTensor(targets)
        self.meta_df_indices = meta_df_indices # These are the original indices from merged_df
    
    def __len__(self): 
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'player_stats': self.player_data[idx],
            'team_features': self.team_data[idx],
            'target': self.targets[idx],
            'original_idx': self.meta_df_indices[idx] # Pass the original DataFrame index
        }

class HybridLSTMNBAModel(nn.Module):
    def __init__(self, n_player_features=12, n_team_features=16,
                 lstm_hidden_size=64, lstm_num_layers=1,
                 player_out_features=128, team_out_features=128, combined_hidden_features=128):
        super().__init__()
        # Player pathway (LSTM)
        self.player_lstm = nn.LSTM(input_size=n_player_features,
                                   hidden_size=lstm_hidden_size,
                                   num_layers=lstm_num_layers,
                                   batch_first=True, # Input: (batch, seq_len, feature_size)
                                   dropout=0.2 if lstm_num_layers > 1 else 0)
        self.player_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, player_out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Team pathway (MLP)
        self.team_net = nn.Sequential(
            nn.Linear(n_team_features, team_out_features),
            nn.BatchNorm1d(team_out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(player_out_features + team_out_features, combined_hidden_features),
            nn.BatchNorm1d(combined_hidden_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_hidden_features, 1)
        )
    
    def forward(self, player_stats, team_features):
        # player_stats: [batch, n_players_padded, n_player_features]
        lstm_out, (h_n, c_n) = self.player_lstm(player_stats)
        # Use the last hidden state of the last LSTM layer
        player_lstm_processed = h_n[-1] # Shape: (batch, lstm_hidden_size)
        player_path_out = self.player_fc(player_lstm_processed)
        
        team_path_out = self.team_net(team_features)
        
        combined_input = torch.cat((player_path_out, team_path_out), dim=1)
        final_output = self.combined_net(combined_input)
        return final_output.squeeze(-1) # Squeeze the last dimension

# ============ TRAINING AND EVALUATION ============
def train_and_evaluate_global(base_path, output_dir_root="Results"):
    """
    Trains the HybridLSTMNBAModel on a global train/test split based on seasons.
    Evaluates and saves rankings for the test set.
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load data
    player_df = load_player_stats(base_path)
    playoff_df = load_playoff_stats(base_path)
    
    # Preprocess data
    # X_player, X_team are features, y_target is Playoff_Rank
    # meta_df is the 'merged_df' containing Team, Season, Playoff_Rank, and original indices
    X_player, X_team, y_target, meta_df = preprocess_data(player_df, playoff_df)

    # Get unique seasons for splitting
    all_seasons_unique = sorted(meta_df['Season'].unique())
    
    if len(all_seasons_unique) < 2:
        print("Not enough unique seasons to perform a train/test split. Need at least 2.")
        return pd.DataFrame()

    # Split seasons into training and testing sets
    train_seasons, test_seasons = train_test_split(all_seasons_unique, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

    print(f"Training seasons: {train_seasons}")
    print(f"Testing seasons: {test_seasons}")

    # Get DataFrame indices for train and test sets based on seasons
    train_df_indices = meta_df[meta_df['Season'].isin(train_seasons)].index.values
    test_df_indices = meta_df[meta_df['Season'].isin(test_seasons)].index.values

    if len(train_df_indices) == 0 or len(test_df_indices) == 0:
        print("Train or test set is empty after season-based split. Check data distribution.")
        return pd.DataFrame()

    # Create full dataset using ALL data, then create Subsets
    # Pass meta_df.index.values as original_indices to NBADataset
    # These indices directly map to rows in X_player, X_team, y_target, and meta_df
    full_dataset = NBADataset(X_player, X_team, y_target, meta_df_indices=meta_df.index.values)
    
    # Create Subsets for training and testing using the DataFrame indices
    # The indices for Subset should be 0-based relative to the full_dataset IF full_dataset was made from np arrays directly.
    # However, since our X_player, X_team, y_target are already ordered as per meta_df,
    # we can use train_df_indices and test_df_indices directly with Subset if they are 0-based indices of these arrays.
    # Let's ensure they are. meta_df.index.values are the actual indices from the DataFrame.
    # We need to map these to 0-based indices for the numpy arrays X_player, X_team, y_target if they are not already.
    # Simpler: filter X_player, X_team, y_target directly.

    X_player_train, X_team_train, y_target_train = X_player[train_df_indices], X_team[train_df_indices], y_target[train_df_indices]
    meta_indices_train = meta_df.iloc[train_df_indices].index.values # Get original DF indices for train set

    X_player_test, X_team_test, y_target_test = X_player[test_df_indices], X_team[test_df_indices], y_target[test_df_indices]
    meta_indices_test = meta_df.iloc[test_df_indices].index.values # Get original DF indices for test set
    
    train_dataset = NBADataset(X_player_train, X_team_train, y_target_train, meta_indices_train)
    test_dataset = NBADataset(X_player_test, X_team_test, y_target_test, meta_indices_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # Increased batch size
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model, optimizer, criterion
    model = HybridLSTMNBAModel(
        n_player_features=X_player.shape[-1], 
        n_team_features=X_team.shape[-1]
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) # Adjusted lr
    criterion = nn.MSELoss() # MSE for rank prediction (treat as regression)
    
    # Training loop
    num_epochs = 100 # Increased epochs
    print("\n=== Starting Global Training ===")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            outputs = model(batch['player_stats'], batch['team_features'])
            loss = criterion(outputs, batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {avg_epoch_loss:.4f}")

    # Evaluation on the test set
    print("\n=== Evaluating on Test Set ===")
    model.eval()
    all_test_predictions = []
    all_test_actuals = []
    all_test_original_indices = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            outputs = model(batch['player_stats'], batch['team_features'])
            all_test_predictions.extend(outputs.cpu().numpy())
            all_test_actuals.extend(batch['target'].cpu().numpy())
            all_test_original_indices.extend(batch['original_idx'].cpu().numpy())

    # Create results DataFrame for the test set
    test_results_df = meta_df.loc[all_test_original_indices].copy() # Get corresponding rows from meta_df
    test_results_df['Predicted_Value'] = all_test_predictions
    test_results_df['Actual_Rank'] = all_test_actuals # This is already the rank

    # Calculate Predicted_Rank based on Predicted_Value (lower value = better rank)
    # Group by season in the test set to rank predictions
    final_ranked_results = []
    for season_in_test in test_results_df['Season'].unique():
        season_data = test_results_df[test_results_df['Season'] == season_in_test].copy()
        # Lower predicted value means better rank
        season_data['Predicted_Rank'] = season_data['Predicted_Value'].rank(method='first', ascending=True).astype(int)
        final_ranked_results.append(season_data)
    
    if not final_ranked_results:
        print("No results to rank from the test set.")
        return pd.DataFrame()

    final_results_output_df = pd.concat(final_ranked_results).sort_values(['Season', 'Predicted_Rank'])
    
    # Save results
    model_specific_output_dir = os.path.join(output_dir_root, "Hybrid_LSTM_Playoff_Ranker")
    os.makedirs(model_specific_output_dir, exist_ok=True)
    output_file_path = os.path.join(model_specific_output_dir, "LSTM_Playoff_Rankings_TestSet.csv")
    
    cols_to_save = ['Season', 'Team', 'Actual_Rank', 'Predicted_Value', 'Predicted_Rank']
    final_results_output_df[cols_to_save].to_csv(output_file_path, index=False)
    print(f"\nTest set rankings saved to: {output_file_path}")
    
    return final_results_output_df[cols_to_save]

if __name__ == "__main__":
    data_base_path = "Preprocessing/Preprocessed Data"
    results_output_root_dir = "Results"
    
    final_test_rankings = train_and_evaluate_global(
        base_path=data_base_path,
        output_dir_root=results_output_root_dir
    )
    
    if not final_test_rankings.empty:
        print("\n=== Test Set Results (Sample) ===")
        print(final_test_rankings.head(20))
    else:
        print("No test results generated.")

