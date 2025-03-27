import pandas as pd
import os 

# create new directory to store data
output_dir = "Preprocessed Data"
os.makedirs(output_dir, exist_ok=True)

# loop through seasons from 2003-04 to 2023-24
for year in range(2003, 2014): # modify this to change what files to filter
    season = f"{year}-{str(year+1)[-2:]}"  # formats as "2003-04", "2004-05", etc.
    input_file = f"Data/{season}.xlsx"  # construct file path
    output_file = f"{output_dir}/{season}_filtered.xlsx"  # output file with "_filtered" suffix

    print(f"Processing {season}...")

    # read both sheets
    regular_df = pd.read_excel(input_file, sheet_name="Regular")
    playoff_df = pd.read_excel(input_file, sheet_name="Playoff")

    # filter regular season dataframe on playoff season dataframe to only keep players who played in the playoffs
    regular_df_filtered = regular_df[
        regular_df.set_index(['Player', 'Team']).index.isin(playoff_df.set_index(['Player', 'Team']).index)
    ]

    # removing "awards" and "player additional" columns
    regular_df_filtered = regular_df_filtered.iloc[:, :-2]

    # checking to see if any entries in playoff_df are not present in filtered df
    missing_players = playoff_df[~playoff_df['Player'].isin(regular_df_filtered['Player'])]
    if not missing_players.empty:
        print(f"Missing players in {season}:")
        print(missing_players)
    else:
        print(f"Preprocessing successful for {season}\n")

    # save to new subdirectory "Preprocessed Data"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        regular_df_filtered.to_excel(writer, sheet_name="Regular", index=False)  # save updated regular sheet
        playoff_df.to_excel(writer, sheet_name="Playoff", index=False)  # save original playoff sheet

    print(f"Saved preprocessed file: {output_file}\n")
