import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import argparse
from tqdm import tqdm
import aiofiles
import asyncio

# Function to generate synthetic data in a batch
def generate_synthetic_data_batch(match_id, num_players, corruption_ratio=0.01, cheater_ratio=0.005):
    player_ids = [str(uuid.uuid4()) for _ in range(num_players)]
    operator_ids = np.random.randint(0, 50, size=num_players)
    
    # Defining approximate kill probabilities by operator and match
    prob_kills = np.array([0.5, 0.25, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01] + [0] * 11)  # 21 elements
    prob_kills /= prob_kills.sum()  # the sum needs be 1

    nb_kills = np.random.choice(range(21), size=num_players, p=prob_kills)

    # Creating cheaters (just for fun)
    cheater_mask = np.random.rand(num_players) < cheater_ratio
    nb_kills[cheater_mask] = np.random.randint(30, 50, size=cheater_mask.sum())  # Cheaters with unusual kills

    # Introduce corrupted lines (as outlined, there is a likelihood of incorrect lines existing)
    corruption_mask = np.random.rand(num_players) < corruption_ratio
    player_ids = np.where(corruption_mask, "", player_ids)
    operator_ids = np.where(corruption_mask, None, operator_ids)
    nb_kills = np.where(corruption_mask, None, nb_kills)

    # Create DataFrame for the batch
    match_data = pd.DataFrame({
        'player_id': player_ids,
        'match_id': match_id,
        'operator_id': operator_ids,
        'nb_kills': nb_kills
    })
    
    return match_data

# Function to generate synthetic data in parallel
def generate_synthetic_data_parallel(num_matches, num_workers=4, players_per_match=10):
    data_batches = []
    
    # create a processing pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_synthetic_data_batch, str(uuid.uuid4()), players_per_match) for _ in range(num_matches)]
        
        # collecting results from each process
        for future in tqdm(futures, total=num_matches, desc="Gerando dados sintéticos"):
            data_batches.append(future.result())

    # Concatenate all batches into a single DataFrame
    return pd.concat(data_batches, ignore_index=True)

# Save the data in file asynchronously
async def save_data_to_log_file(data):
    current_date = datetime.now().strftime("%Y%m%d")
    file_name = f"r6-matches-{current_date}.log"
    async with aiofiles.open(file_name, mode='w') as file:
        await file.write(data.to_csv(index=False, header=False))
    print(f"Log file saved as: {file_name}")

# Main Function
async def main(num_matches=1000000, num_workers=4):
    # num_matches = 30000000  # number of matches
    # num_workers = 4  # number of workers
    
    # generate data in parallel process
    data = generate_synthetic_data_parallel(num_matches, num_workers=num_workers)
    
    # save the data
    await save_data_to_log_file(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for Rainbow Six Siege matches.")
    parser.add_argument("--num_matches", type=int, default=1000000, help="Number of matches to be generated.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes to be used.")
    
    args = parser.parse_args()
    
    # main(num_matches=args.num_matches, num_workers=args.num_workers)
    asyncio.run(main(num_matches=args.num_matches, num_workers=args.num_workers))