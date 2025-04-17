import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata # Moved import to top
from tqdm import tqdm
import traceback # For detailed error printing

# ============ CONFIGURATION ============
# Dictionary to map team abbreviations and variations to a standard name
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
    # Allow mapping full names to themselves to handle potential inconsistencies
    'Atlanta Hawks': 'Atlanta Hawks', 'Boston Celtics': 'Boston Celtics', 'Brooklyn Nets': 'Brooklyn Nets',
    'Charlotte Hornets': 'Charlotte Hornets', 'Chicago Bulls': 'Chicago Bulls', 'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Dallas Mavericks': 'Dallas Mavericks', 'Denver Nuggets': 'Denver Nuggets', 'Detroit Pistons': 'Detroit Pistons',
    'Golden State Warriors': 'Golden State Warriors', 'Houston Rockets': 'Houston Rockets', 'Indiana Pacers': 'Indiana Pacers',
    'Los Angeles Clippers': 'Los Angeles Clippers', 'Los Angeles Lakers': 'Los Angeles Lakers', 'Memphis Grizzlies': 'Memphis Grizzlies',
    'Miami Heat': 'Miami Heat', 'Milwaukee Bucks': 'Milwaukee Bucks', 'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'New Orleans Pelicans': 'New Orleans Pelicans', 'New York Knicks': 'New York Knicks', 'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Orlando Magic': 'Orlando Magic', 'Philadelphia 76ers': 'Philadelphia 76ers', 'Phoenix Suns': 'Phoenix Suns',
    'Portland Trail Blazers': 'Portland Trail Blazers', 'Sacramento Kings': 'Sacramento Kings', 'San Antonio Spurs': 'San Antonio Spurs',
    'Toronto Raptors': 'Toronto Raptors', 'Utah Jazz': 'Utah Jazz', 'Washington Wizards': 'Washington Wizards',
    # Handle older names if necessary (add more as needed)
    'New Jersey Nets': 'Brooklyn Nets', # Example mapping
    'Charlotte Bobcats': 'Charlotte Hornets' # Example mapping
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
    """
    Preprocesses player and playoff data to create features and targets for the model.

    Args:
        player_df (pd.DataFrame): DataFrame from load_player_stats.
        playoff_df (pd.DataFrame): DataFrame from load_playoff_stats.

    Returns:
        tuple: Contains:
            - np.ndarray: Scaled player statistics arrays (padded).
            - np.ndarray: Scaled aggregated team features.
            - np.ndarray: Playoff ranks (targets), as float32.
            - pd.DataFrame: Metadata (Team, Season, original rank) for mapping results.
    """
    print("Starting preprocessing...")

    # --- Team Aggregation ---
    # Define a safe weighted average function
    def weighted_avg(values, weights):
        values = pd.to_numeric(values, errors='coerce')
        weights = pd.to_numeric(weights, errors='coerce')
        mask = (weights > 0) & (~np.isnan(values)) & (~np.isnan(weights)) # Ensure weights > 0 and values/weights are numeric
        if mask.any():
            return np.average(values[mask], weights=weights[mask])
        return 0.0 # Return float 0.0

    # Define aggregation functions including top 3 logic
    agg_funcs = {
        'MP': 'sum',
        'FG%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        '3P%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'eFG%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'FT%': lambda x: weighted_avg(x, player_df.loc[x.index, 'G']),
        'TRB': 'sum',
        # Calculate sum and mean of top 3 AST/PTS based on non-missing values
        'AST': ['sum', lambda x: x.dropna().nlargest(3).mean() if len(x.dropna()) >= 3 else 0.0],
        'STL': 'sum',
        'BLK': 'sum',
        'TOV': 'sum',
        'PF': 'sum',
        'PTS': ['sum', lambda x: x.dropna().nlargest(3).mean() if len(x.dropna()) >= 3 else 0.0],
        # Calculate mean and standard deviation of games played (use population std dev, ddof=0)
        'G': ['mean', lambda x: x.dropna().std(ddof=0) if len(x.dropna()) > 0 else 0.0]
    }

    # Apply aggregation
    print("Aggregating player stats to team level...")
    team_agg = player_df.groupby(['Team', 'Season']).agg(agg_funcs).reset_index()

    # Flatten multi-index columns resulting from aggregation
    team_agg.columns = [
        'Team', 'Season', 'MP_sum', 'FG%_wt', '3P%_wt', 'eFG%_wt', 'FT%_wt',
        'TRB_sum', 'AST_sum', 'AST_top3', 'STL_sum', 'BLK_sum', 'TOV_sum',
        'PF_sum', 'PTS_sum', 'PTS_top3', 'G_mean', 'G_std'
    ]

    # Fill any NaNs that might have resulted from aggregation (e.g., std dev of single value)
    team_agg = team_agg.fillna(0.0)
    print(f"Aggregated team stats shape: {team_agg.shape}")

    # --- Merge with Playoff Data ---
    print("Merging team stats with playoff ranks...")
    # Ensure target column is present and numeric before merge
    if 'Playoff_Rank' not in playoff_df.columns:
         raise ValueError("Playoff DataFrame is missing the 'Playoff_Rank' column.")
    playoff_df['Playoff_Rank'] = pd.to_numeric(playoff_df['Playoff_Rank'], errors='coerce')
    playoff_df = playoff_df.dropna(subset=['Playoff_Rank']) # Drop teams without valid rank
    playoff_df['Playoff_Rank'] = playoff_df['Playoff_Rank'].astype(np.float32) # Ensure float32 for PyTorch loss

    merged = pd.merge(team_agg, playoff_df, on=['Team', 'Season'], how='inner') # Inner merge keeps only teams present in both
    print(f"Merged data shape (teams with both stats and playoff rank): {merged.shape}")
    if merged.empty:
        raise ValueError("No teams found matching between aggregated stats and playoff ranks. Check team names and seasons.")

    # --- Prepare Player Arrays (Sequence Data) ---
    print("Preparing player arrays (sequences)...")
    player_cols = ['MP', 'FG%', '3P%', 'eFG%', 'FT%', 'TRB', 'AST',
                   'STL', 'BLK', 'TOV', 'PF', 'PTS'] # Features per player
    player_arrays = []
    valid_indices_for_merged = [] # Keep track of indices in 'merged' that have valid player data

    for i, row in merged.iterrows():
        team_players = player_df[(player_df['Team'] == row['Team']) &
                               (player_df['Season'] == row['Season'])]

        if team_players.empty:
            # print(f"Warning: No player data found for {row['Team']} in season {row['Season']}. Skipping this team.") # Optional warning
            continue # Skip this row in merged_df

        # Select columns, convert to numeric (coerce errors), fill NaNs, convert to numpy
        arr = team_players[player_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        player_arrays.append(arr)
        valid_indices_for_merged.append(i) # Store the index 'i' from the original merged df

    # Filter the merged dataframe to only include rows for which we found player data
    if not valid_indices_for_merged:
         raise ValueError("No teams remaining after matching player data. Check player_df content.")
    merged = merged.iloc[valid_indices_for_merged].reset_index(drop=True)
    print(f"Shape after ensuring player data exists: {merged.shape}")

    # Pad player arrays to have the same sequence length (max number of players on any team)
    if not player_arrays: # Should not happen if valid_indices check passed, but good practice
         raise ValueError("Player arrays list is empty after filtering.")
    max_players = max(arr.shape[0] for arr in player_arrays)
    n_player_features = player_arrays[0].shape[1] # Get number of features from the first array
    print(f"Padding player sequences to max length: {max_players}")

    # Create a zero array for padding
    player_arrays_padded = np.zeros((len(player_arrays), max_players, n_player_features), dtype=np.float32)
    for idx, arr in enumerate(player_arrays):
        player_arrays_padded[idx, :arr.shape[0], :] = arr # Fill with actual data

    # --- Scale Features ---
    # Scale player stats (scale across all players and features)
    print("Scaling player features...")
    player_scaler = StandardScaler()
    # Reshape for scaler: [n_teams * max_players, n_features]
    original_shape = player_arrays_padded.shape
    player_arrays_reshaped = player_arrays_padded.reshape(-1, n_player_features)
    # Fit and transform, handling potential NaNs/Infs just in case
    player_arrays_reshaped = np.nan_to_num(player_arrays_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
    # Avoid scaling if variance is zero (e.g., all zeros after padding)
    if np.all(np.std(player_arrays_reshaped, axis=0) < 1e-6):
         print("Warning: Player data has near-zero variance. Skipping scaling.")
         player_arrays_scaled = player_arrays_reshaped
    else:
         player_arrays_scaled = player_scaler.fit_transform(player_arrays_reshaped)
    # Reshape back to [n_teams, max_players, n_features]
    X_player = player_arrays_scaled.reshape(original_shape)

    # Scale aggregated team features
    print("Scaling team features...")
    team_feature_cols = [col for col in team_agg.columns if col not in ['Team', 'Season']]
    team_features_df = merged[team_feature_cols]
    # Convert to numeric, fill NaNs, get values
    team_features_raw = team_features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values

    team_scaler = StandardScaler()
    # Fit and transform, handling potential NaNs/Infs
    team_features_raw = np.nan_to_num(team_features_raw, nan=0.0, posinf=0.0, neginf=0.0)
     # Avoid scaling if variance is zero
    if np.all(np.std(team_features_raw, axis=0) < 1e-6):
        print("Warning: Team data has near-zero variance. Skipping scaling.")
        X_team = team_features_raw.astype(np.float32)
    else:
        X_team = team_scaler.fit_transform(team_features_raw).astype(np.float32)

    # Prepare targets (ensure float32)
    y = merged['Playoff_Rank'].values.astype(np.float32)

    # --- Final Validation ---
    print("Final validation of processed data...")
    assert not np.isnan(X_player).any(), "NaN values found in final player features"
    assert not np.isinf(X_player).any(), "Inf values found in final player features"
    assert not np.isnan(X_team).any(), "NaN values found in final team features"
    assert not np.isinf(X_team).any(), "Inf values found in final team features"
    assert not np.isnan(y).any(), "NaN values found in final targets"
    assert not np.isinf(y).any(), "Inf values found in final targets"
    assert X_player.shape[0] == X_team.shape[0] == y.shape[0], \
        f"Mismatch in final data shapes: Player({X_player.shape[0]}), Team({X_team.shape[0]}), Target({y.shape[0]})"

    print("Preprocessing finished successfully.")
    # Return the final features, target, and the filtered metadata dataframe
    return X_player, X_team, y, merged[['Team', 'Season', 'Playoff_Rank']]


# ============ NEURAL NETWORK (RNN VERSION) ============
class NBADataset(Dataset):
    """
    PyTorch Dataset class for NBA data.
    """
    def __init__(self, player_arrays, team_features, targets, original_indices=None):
        # Ensure data is torch tensors of the correct type
        self.player_data = torch.tensor(player_arrays, dtype=torch.float32)
        self.team_data = torch.tensor(team_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) # Target should be float for regression/MSELoss

        # Store original indices if provided (useful for tracking during train/test splits)
        self.original_indices = original_indices if original_indices is not None else np.arange(len(targets))

        # Basic validation
        assert len(self.player_data) == len(self.team_data) == len(self.targets), "Data length mismatch in Dataset"

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing player stats, team features, target, and original index.
        """
        if idx >= len(self):
             raise IndexError(f"Index {idx} out of bounds for dataset length {len(self)}")

        # Get the original index corresponding to this dataset item
        # This is important when using subsets, as 'idx' refers to the subset index
        original_idx = self.original_indices[idx] if self.original_indices is not None else idx

        return {
            'player_stats': self.player_data[idx],
            'team_features': self.team_data[idx],
            'target': self.targets[idx],
            'original_idx': original_idx # Return the index from the *original* full dataset
        }

class RNNNBAModel(nn.Module):
    """
    RNN-based model for NBA playoff rank prediction.
    Uses an LSTM for player sequences and an MLP for team features.
    """
    def __init__(self, n_player_features, n_team_features, rnn_hidden_size=64, num_rnn_layers=1, dropout_rate=0.3):
        """
        Initializes the RNNNBAModel.

        Args:
            n_player_features (int): Number of features per player (e.g., PTS, AST).
            n_team_features (int): Number of aggregated team features.
            rnn_hidden_size (int): Number of features in the LSTM hidden state.
            num_rnn_layers (int): Number of recurrent layers in the LSTM.
            dropout_rate (float): Dropout probability for regularization.
        """
        super().__init__()
        self.n_player_features = n_player_features
        self.n_team_features = n_team_features
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_layers = num_rnn_layers

        # --- Player Pathway (LSTM) ---
        # batch_first=True expects input shape [batch, seq_len (players), features]
        self.player_rnn = nn.LSTM(
            input_size=n_player_features,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            # Apply dropout between LSTM layers only if num_layers > 1
            dropout=dropout_rate if num_rnn_layers > 1 else 0.0
        )
        # Dropout layer applied to the final output of the RNN sequence processing
        self.player_dropout_after_rnn = nn.Dropout(dropout_rate)

        # --- Team Pathway (MLP) ---
        self.team_net = nn.Sequential(
            nn.Linear(n_team_features, 128), # Input layer
            nn.BatchNorm1d(128),             # Batch normalization for stability
            nn.ReLU(),                       # Activation function
            nn.Dropout(dropout_rate)         # Dropout for regularization
        )

        # --- Combined Network ---
        # The input size is the sum of the outputs from the player and team pathways
        combined_input_size = rnn_hidden_size + 128 # (LSTM hidden size + Team MLP output size)
        self.combined = nn.Sequential(
            nn.Linear(combined_input_size, 128), # Hidden layer
            nn.BatchNorm1d(128),                # Batch normalization
            nn.ReLU(),                          # Activation function
            nn.Dropout(dropout_rate),           # Dropout
            nn.Linear(128, 1)                   # Final output layer (predicting a single value)
        )

    def forward(self, player_stats, team_features):
        """
        Defines the forward pass of the model.

        Args:
            player_stats (torch.Tensor): Player statistics tensor shape [batch, seq_len (players), n_player_features].
            team_features (torch.Tensor): Team features tensor shape [batch, n_team_features].

        Returns:
            torch.Tensor: The predicted output tensor shape [batch].
        """
        # --- Process Player Stats through RNN ---
        # player_stats shape: [batch, players, features]
        # We need the RNN's output, typically the last hidden state.
        # The LSTM returns: output, (h_n, c_n)
        #   output: contains the output features (h_t) from the last layer of the LSTM, for each t. Shape [batch, seq_len, hidden_size]
        #   h_n:    contains the final hidden state for each element in the batch. Shape [num_layers * num_directions, batch, hidden_size]
        #   c_n:    contains the final cell state for each element in the batch. Shape [num_layers * num_directions, batch, hidden_size]
        _, (h_n, _) = self.player_rnn(player_stats) # We only need the hidden state h_n

        # h_n contains hidden states for all layers. We usually take the hidden state
        # from the last layer. If num_layers=1, h_n[0] is the one. If num_layers>1, h_n[-1] is the last layer's state.
        # Shape of h_n[-1]: [batch, hidden_size]
        player_out = h_n[-1]

        # Apply dropout after getting the final RNN hidden state representation
        player_out = self.player_dropout_after_rnn(player_out)

        # --- Process Team Features through MLP ---
        # team_features shape: [batch, n_team_features]
        team_out = self.team_net(team_features) # Output shape: [batch, 128]

        # --- Combine Pathways ---
        # Concatenate the outputs along the feature dimension (dim=1)
        combined_features = torch.cat([player_out, team_out], dim=1) # Shape: [batch, rnn_hidden_size + 128]

        # --- Final Prediction ---
        prediction = self.combined(combined_features) # Shape: [batch, 1]

        # Remove the last dimension (dimension 1) to get shape [batch]
        # This is important for compatibility with loss functions like MSELoss expecting [batch] vs [batch].
        prediction = prediction.squeeze(-1)

        return prediction


# ============ CUSTOM SUBSET CLASS (for tracking original indices) ============
class NBASubset(torch.utils.data.Subset):
    """
    A custom Subset class that correctly retrieves items using the original dataset's
    __getitem__ logic and preserves the original index.
    """
    def __getitem__(self, idx):
        """
        Retrieves an item from the subset.

        Args:
            idx (int): The index within this subset.

        Returns:
            dict: The item dictionary obtained from the original dataset,
                  including the correct 'original_idx'.
        """
        # self.indices[idx] gives the index in the *original* dataset
        original_dataset_index = self.indices[idx]
        # Call the original dataset's __getitem__ with the original index
        return self.dataset[original_dataset_index]


# ============ TRAINING AND EVALUATION ============
def train_and_evaluate(base_path, output_dir="Results_RNN", num_epochs=50, batch_size=8, learning_rate=0.001):
    """
    Trains the RNNNBAModel and evaluates it season by season.

    Args:
        base_path (str): Path to the directory containing the data subfolders.
        output_dir (str): Directory to save the results CSV.
        num_epochs (int): Number of training epochs per season.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        pd.DataFrame or None: DataFrame containing the combined results across all seasons,
                             or None if an error occurs.
    """
    # --- Load and Preprocess Data ---
    try:
        print(f"Loading data from base path: {base_path}")
        player_df = load_player_stats(base_path)
        playoff_df = load_playoff_stats(base_path)
        X_player, X_team, y, meta = preprocess_data(player_df, playoff_df)
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Error during data loading or preprocessing: {e}")
        print("Please check the data paths and file contents.")
        # traceback.print_exc() # Uncomment for detailed traceback
        return None

    # --- Create Full Dataset ---
    try:
        # Pass original indices from the 'meta' DataFrame's index
        full_dataset = NBADataset(X_player, X_team, y, original_indices=meta.index.values)
        if len(full_dataset) == 0:
             print("Error: Full dataset is empty after creation.")
             return None
        print(f"Full dataset created with {len(full_dataset)} samples.")
    except Exception as e:
        print(f"Error creating NBADataset: {e}")
        # traceback.print_exc()
        return None

    # --- Setup Device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Add a check here in case CUDA is detected but unusable
    if device == torch.device("cuda"):
        try:
            # Try a simple CUDA operation to confirm it's working
            _ = torch.tensor([1.0, 2.0]).to(device)
            print("CUDA device confirmed.")
        except RuntimeError as e:
            print(f"CUDA detected but unusable: {e}. Falling back to CPU.")
            device = torch.device("cpu")
            print(f"Now using device: {device}")


    # --- Process Season by Season ---
    seasons = sorted(meta['Season'].unique())
    all_results = []

    for season in seasons:
        print(f"\n=== Processing Season: {season} ===")

        # Find indices in the original 'meta' DataFrame belonging to this season
        season_original_indices = meta[meta['Season'] == season].index.values

        if len(season_original_indices) < 4: # Need enough data for a meaningful split
            print(f"Skipping season {season} - only {len(season_original_indices)} teams available (minimum 4 required for split).")
            continue

        # --- Create Season-Specific Dataset View using NBASubset ---
        # NBASubset takes the full_dataset and the list of original indices for this season
        try:
            season_dataset_view = NBASubset(full_dataset, season_original_indices)
            print(f"Created dataset view for season {season} with {len(season_dataset_view)} samples.")
        except Exception as e:
             print(f"Error creating subset view for season {season}: {e}")
             continue

        # --- Split Season Data into Train/Test ---
        total_season_samples = len(season_dataset_view)
        # Ensure at least 1 sample in test set if possible
        train_size = max(1, int(0.8 * total_season_samples))
        test_size = total_season_samples - train_size
        if test_size == 0 and train_size > 1:
            train_size -= 1
            test_size += 1
        elif test_size == 0 and train_size <=1:
             print(f"Skipping season {season} - cannot create train/test split with {total_season_samples} samples.")
             continue

        try:
            train_set, test_set = torch.utils.data.random_split(
                season_dataset_view,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(42) # Reproducible splits
            )
            print(f"Split season {season} into {len(train_set)} train / {len(test_set)} test samples.")
        except Exception as e:
             print(f"Error splitting data for season {season}: {e}")
             continue

        # --- Create DataLoaders ---
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for simplicity/debugging
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        # Loader for evaluating the full season data at the end
        full_season_loader = DataLoader(season_dataset_view, batch_size=batch_size, shuffle=False, num_workers=0)

        # --- Initialize Model, Optimizer, Loss ---
        # Determine feature dimensions from the data
        n_player_features = X_player.shape[2]
        n_team_features = X_team.shape[1]

        model = RNNNBAModel(
             n_player_features=n_player_features,
             n_team_features=n_team_features,
             rnn_hidden_size=64,   # Hyperparameter: LSTM hidden size
             num_rnn_layers=2,     # Hyperparameter: Number of LSTM layers
             dropout_rate=0.4      # Hyperparameter: Dropout rate
        ).to(device) # Move model to the selected device (GPU/CPU)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Adam optimizer with L2 regularization
        criterion = nn.MSELoss() # Mean Squared Error loss for regression/ranking proxy

        # --- Training Loop ---
        print(f"Starting training for season {season} ({num_epochs} epochs)...")
        for epoch in range(num_epochs):
            model.train() # Set model to training mode
            epoch_loss = 0.0
            num_batches = 0
            # Use tqdm for progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
            for batch in pbar:
                # Move batch data to the device
                player_stats = batch['player_stats'].to(device)
                team_features = batch['team_features'].to(device)
                targets = batch['target'].to(device)

                # Forward pass
                optimizer.zero_grad() # Zero gradients before calculation
                outputs = model(player_stats, team_features)

                # Ensure shapes match for loss calculation (output: [batch], target: [batch])
                if outputs.shape != targets.shape:
                     try:
                         outputs = outputs.view_as(targets)
                     except RuntimeError as reshape_err:
                         print(f"\nShape mismatch error: Output {outputs.shape}, Target {targets.shape}. Skipping batch. Error: {reshape_err}")
                         continue # Skip batch if reshape fails

                # Calculate loss
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward() # Compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to prevent explosion
                optimizer.step() # Update model weights

                epoch_loss += loss.item()
                num_batches += 1
                # Corrected line: Format the loss value using an f-string
                pbar.set_postfix({'loss': f"{loss.item():.4f}"}) # Update progress bar description

            # Print average loss for the epoch
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                # Only print epoch loss every few epochs or at the end to reduce clutter
                if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                    print(f"Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {avg_epoch_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} - No batches processed in training loader.")


        # --- Evaluation for the Season ---
        print(f"Evaluating model on full data for season {season}...")
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        all_season_preds = []
        all_season_targets = []
        all_season_original_indices = [] # Store original indices corresponding to predictions

        with torch.no_grad(): # Disable gradient calculations for evaluation
            for batch in full_season_loader:
                # Move data to device
                player_stats = batch['player_stats'].to(device)
                team_features = batch['team_features'].to(device)

                # Get predictions
                predictions = model(player_stats, team_features)

                # Store predictions, targets, and original indices (move predictions back to CPU)
                all_season_preds.extend(predictions.cpu().numpy())
                all_season_targets.extend(batch['target'].numpy()) # Targets are already on CPU from DataLoader
                all_season_original_indices.extend(batch['original_idx'].numpy()) # Original indices too

        # --- Process Season Results ---
        if not all_season_preds:
             print(f"Warning: No predictions generated during evaluation for season {season}. Skipping results.")
             continue

        # Wrap ranking and DataFrame creation in try-except
        try:
            # Create results dataframe using the collected original indices to fetch correct Team/Season info
            # Map these original indices back to the 'meta' dataframe
            # Ensure indices are unique and sorted for consistent mapping
            unique_indices, order = np.unique(all_season_original_indices, return_index=True)
            if len(unique_indices) == 0: # Check if filtering resulted in empty data
                print(f"Warning: No unique indices found after evaluation for season {season}. Skipping results.")
                continue
            season_meta_filtered = meta.loc[unique_indices] # Get corresponding rows from original metadata

            # Align predictions and targets with the unique indices order
            preds_aligned = np.array(all_season_preds)[order]
            targets_aligned = np.array(all_season_targets)[order]

            # Check if arrays are empty before ranking
            if len(preds_aligned) == 0 or len(targets_aligned) == 0:
                 print(f"Warning: Empty prediction or target array after alignment for season {season}. Skipping results.")
                 continue

            # Calculate predicted ranks based on model output (lower score = better rank assumed)
            predicted_ranks = rankdata(preds_aligned, method='ordinal') # 'ordinal' assigns unique ranks 1, 2, 3...

            # Calculate actual ranks based on the ground truth targets
            actual_ranks = rankdata(targets_aligned, method='ordinal')

            # Create DataFrame for this season's results
            season_results = pd.DataFrame({
                'Season': season,
                'Team': season_meta_filtered['Team'].tolist(), # Get team names from filtered meta
                'Actual_Rank': actual_ranks.astype(int),
                'Predicted_Rank': predicted_ranks.astype(int),
                'Raw_Prediction': preds_aligned, # Include raw score for inspection
                'Original_Index': unique_indices # Store the original index from 'meta'
            }).sort_values('Predicted_Rank') # Sort results by predicted rank

            print(f"\n=== Season {season} Rankings (RNN Model) ===")
            # Display sorted results clearly
            print(season_results[['Predicted_Rank', 'Team', 'Actual_Rank']]
                  .rename(columns={'Predicted_Rank': 'Pred_Rank', 'Actual_Rank': 'Actual'})
                  .to_string(index=False))

            all_results.append(season_results) # Add season results to the list

        except ImportError:
             print("\nError: 'scipy' library not found. Please install it (`pip install scipy`) to use rankdata.")
             print("Skipping ranking and results generation for this season and subsequent ones.")
             return None # Stop processing if scipy is missing
        except Exception as e:
            print(f"\nError occurred during ranking or DataFrame creation for season {season}: {e}")
            print("Skipping results for this season.")
            traceback.print_exc() # Print detailed traceback for debugging
            continue # Continue to the next season


    # --- Final Aggregation and Saving ---
    if not all_results:
        print("\nNo results were generated for any season. Cannot save output.")
        return None

    print("\nConcatenating results from all processed seasons...")
    try:
        final_results = pd.concat(all_results, ignore_index=True)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "RNN_complete_rankings.csv")

        # Select and order columns for the final CSV
        final_results.to_csv(output_path,
                           columns=['Season', 'Predicted_Rank', 'Team', 'Actual_Rank', 'Raw_Prediction', 'Original_Index'],
                           index=False)
        print(f"\nAll seasons rankings saved successfully to: {output_path}")
        return final_results

    except Exception as e:
        print(f"\nError saving final results to CSV: {e}")
        # traceback.print_exc()
        return None


# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    # Define the base path where 'Player Stats...' and 'Actual Playoff...' folders are located
    # IMPORTANT: Adjust this path relative to where you run the script, or use an absolute path.
    data_path = "Preprocessing/Preprocessed Data"

    print(f"Script starting. Looking for data in: {os.path.abspath(data_path)}")

    # Check if the base data directory exists
    if not os.path.isdir(data_path):
         print(f"\nError: Base data directory not found at '{data_path}'.")
         print("Please ensure the path is correct and contains the necessary subdirectories:")
         print(f"- {os.path.join(data_path, 'Player Stats Regular and Playoff')}")
         print(f"- {os.path.join(data_path, 'Actual Playoff Team Stats')}")
    else:
        # Check for subdirectories
        player_subdir = os.path.join(data_path, 'Player Stats Regular and Playoff')
        playoff_subdir = os.path.join(data_path, 'Actual Playoff Team Stats')
        if not os.path.isdir(player_subdir):
             print(f"Error: Player stats subdirectory not found: '{player_subdir}'")
        elif not os.path.isdir(playoff_subdir):
             print(f"Error: Playoff stats subdirectory not found: '{playoff_subdir}'")
        else:
             # Check if SciPy is available before starting
             try:
                 from scipy.stats import rankdata
                 print("SciPy library found.")
                 scipy_available = True
             except ImportError:
                 print("\nWarning: 'scipy' library not found. Ranking functionality will be disabled.")
                 print("Install it using: pip install scipy")
                 scipy_available = False # Allow script to run, but ranking part will fail gracefully later

             print("Data directories found. Starting training and evaluation...")
             # --- Run the main function ---
             results_df = train_and_evaluate(
                 base_path=data_path,
                 output_dir="Results/RNN_v2", # Changed output dir slightly
                 num_epochs=50,             # Example: Number of epochs
                 batch_size=8,              # Example: Batch size
                 learning_rate=0.001        # Example: Learning rate
             )

             # --- Print Results Preview ---
             if results_df is not None:
                 print("\n--- Final Results Preview (RNN Model) ---")
                 print(results_df.head(20)) # Show top 20 rows of the combined results
             else:
                 print("\nTraining and evaluation process failed or produced no results.")

    print("\nScript finished.")
