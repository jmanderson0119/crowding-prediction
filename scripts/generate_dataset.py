import pandas as pd
import os
import random
from scipy.stats import pearsonr

def extract_return_by_id(returns: pd.DataFrame, id: int) -> list: return [round(x, 6) for x in returns[returns["PERMNO"] == id]["resret"].tolist()]

def tabulate_returns(ranks: pd.DataFrame, returns: pd.DataFrame) -> tuple:
    # preprocess the 'resret' column
    returns['resret'] = returns['resret'].str.rstrip('%').astype('float') / 100.0
    
    # get the latest rankmom for each permno
    latest_ranks = ranks.sort_values(['PERMNO', 'DATE'], ascending=[True, False]).groupby('PERMNO').first().reset_index()
    rank_dict = dict(zip(latest_ranks['PERMNO'], latest_ranks['rankmom']))
    
    # get unique permnos from returns that are present in the rank_dict
    all_ids = returns['PERMNO'].unique().tolist()
    
    # split permnos into losers (rankmom=1) and winners (rankmom=10)
    loser_ids = [id for id in all_ids if rank_dict.get(id, None) == 1]
    winner_ids = [id for id in all_ids if rank_dict.get(id, None) == 10]
    
    # function to create dataframe for a list of ids
    def create_returns_df(ids):
        return pd.DataFrame([extract_return_by_id(returns, i) for i in ids])
    
    # generate the two dataframes
    loser_returns = create_returns_df(loser_ids)
    winner_returns = create_returns_df(winner_ids)
    
    return loser_returns, winner_returns

def extract_most_recent_period(returns_table: pd.DataFrame) -> pd.DataFrame: return returns_table.iloc[:, -252:].dropna()

def generate_stock_pairs(df: pd.DataFrame, num_pairs: int) -> pd.DataFrame:
    n = len(df)
    if n < 2: raise ValueError("DataFrame must have at least 2 rows to form pairs.")
    
    max_pairs = n * (n - 1) // 2
    if num_pairs > max_pairs: raise ValueError(f"Requested {num_pairs} pairs exceeds maximum possible {max_pairs} pairs.")
    
    used_pairs = set()
    result = []
    
    while len(result) < num_pairs:
        i, j = random.sample(range(n), 2)
        pair = tuple(sorted((i, j)))
        
        if pair not in used_pairs:
            used_pairs.add(pair)
            row1 = [round(x, 6) for x in df.iloc[i].tolist()]
            row2 = [round(x, 6) for x in df.iloc[j].tolist()]
            
            corr, _ = pearsonr(row1, row2)
            rounded_corr = round(corr, 6)
            
            result.append({
                'stock1': row1,
                'stock2': row2,
                'correlated': 0 if rounded_corr >= 0.3 else 1
            })
    
    return pd.DataFrame(result)

def main():
    random.seed(42)
    
    # load and process data
    ranks = pd.read_csv("../data/raw_data/ranks.csv")
    returns = pd.read_csv("../data/raw_data/returns.csv")
    loser_returns, winner_returns = tabulate_returns(ranks, returns)
    
    # save processed returns
    os.makedirs("../data/lifetime_returns_data", exist_ok=True)
    loser_returns.to_csv("../data/lifetime_returns_data/tabulated_loser_returns.csv", index=False)
    winner_returns.to_csv("../data/lifetime_returns_data/tabulated_winner_returns.csv", index=False)
    
    # extract most recent trading period
    recent_loser_returns = extract_most_recent_period(loser_returns)
    recent_winner_returns = extract_most_recent_period(winner_returns)
    
    # save recent returns
    os.makedirs("../data/recent_trading_period_returns_data", exist_ok=True)
    recent_loser_returns.to_csv("../data/recent_trading_period_returns_data/recent_loser_returns.csv", index=False)
    recent_winner_returns.to_csv("../data/recent_trading_period_returns_data/recent_winner_returns.csv", index=False)
    
    # split into train and test sets
    loser_train = recent_loser_returns.sample(frac=0.7, random_state=42)  # Add random_state here
    loser_test = recent_loser_returns.drop(loser_train.index)
    winner_train = recent_winner_returns.sample(frac=0.7, random_state=42)  # Add random_state here
    winner_test = recent_winner_returns.drop(winner_train.index)
    
    # generate training pairs from train subsets
    train_loser_pairs = generate_stock_pairs(loser_train, len(loser_train) * (len(loser_train) - 1) // 2)
    train_winner_pairs = generate_stock_pairs(winner_train, len(winner_train) * (len(winner_train) - 1) // 2)
    
    # generate test pairs from test subsets
    test_loser_pairs = generate_stock_pairs(loser_test, len(loser_test) * (len(loser_test) - 1) // 2)
    test_winner_pairs = generate_stock_pairs(winner_test, len(winner_test) * (len(winner_test) - 1) // 2)
    
    # combine training and test datasets
    training_pairs = pd.concat([train_loser_pairs, train_winner_pairs], ignore_index=True)
    test_pairs = pd.concat([test_loser_pairs, test_winner_pairs], ignore_index=True)
    
    # save final processed data
    os.makedirs("../data/processed_data", exist_ok=True)
    training_pairs.to_csv("../data/processed_data/training_pairs.csv", index=False)
    test_pairs.to_csv("../data/processed_data/test_pairs.csv", index=False)

if __name__ == "__main__": main()