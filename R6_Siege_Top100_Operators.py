import sys
import pandas as pd
import numpy as np
import uuid
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import psutil
import threading
import time
import aiofiles
import asyncio

# Dimensionality reduction using PCA
def reduce_dimensions(data):
    # Filter data that does not have NaN
    data = data.dropna(subset=['nb_kills']).copy()
    if data.empty:
        return data  # Return an empty DataFrame if all data is NaN
    pca = PCA(n_components=1)  # We reduce to 1 dimension
    # data['pca_1'] = pca.fit_transform(data[['nb_kills']])
    data.loc[:, 'pca_1'] = pca.fit_transform(data[['nb_kills']])
    return data

# # Intelligent grouping of data using KMeans
# def cluster_data(data):
#     kmeans = KMeans(n_clusters=5, random_state=73)  # 5 clusters
#     # data['cluster'] = kmeans.fit_predict(data[['nb_kills']])
#     data.loc[:, 'cluster'] = kmeans.fit_predict(data[['nb_kills']])
#     return data

# Intelligent grouping of data using KMeans
def cluster_data(data):
    if len(data) < 5:  # If there are fewer than 5 samples
        data.loc[:, 'cluster'] = -1  # Assign a default cluster or handle accordingly
        return data
    kmeans = KMeans(n_clusters=min(5, len(data)), random_state=73)  # Use min(5, len(data))
    data.loc[:, 'cluster'] = kmeans.fit_predict(data[['nb_kills']])
    return data

# Anomaly detection using Isolation Forest
def detect_anomalies(data):
    iso_forest = IsolationForest(contamination=0.01, random_state=73)  # 1% expected contamination
    # data['is_anomaly'] = iso_forest.fit_predict(data[['nb_kills']])
    data.loc[:, 'is_anomaly'] = iso_forest.fit_predict(data[['nb_kills']])
    return data[data['is_anomaly'] == 1]  # We only keep normal data

# Function to process data chunks in parallel
def process_chunk(data_chunk):
    # Applying dimensionality reduction
    reduced_data = reduce_dimensions(data_chunk)
    if reduced_data.empty:
        return reduced_data  # Returns an empty DataFrame if there is no valid data
    # Grouping data by kills
    clustered_data = cluster_data(reduced_data)
    # Detecting anomalies and filtering
    clean_data = detect_anomalies(clustered_data)
    return clean_data

# Function to calculate the top 100 operators
def calculate_operator_top100(df):
    # Group by operator and match and calculate average kills
    operator_stats = df.groupby(['operator_id', 'match_id'])['nb_kills'].mean().reset_index()
    
    # Sort by average kills and take the first 100 operators
    top_100 = operator_stats.groupby('operator_id')['nb_kills'].mean().nlargest(100).reset_index()
    
    # For each operator, list their average kills in different matches
    top_100_details = []
    for operator_id in top_100['operator_id']:
        operator_matches = operator_stats[operator_stats['operator_id'] == operator_id]
        match_list = ",".join([f"{row.match_id}:{round(row.nb_kills, 2)}" for row in operator_matches.itertuples()])
        top_100_details.append(f"{operator_id}|{match_list}")
    
    return top_100_details

# Function to save data to text file asynchronously
async def save_operator_top100(data, date):
    filename = f"operator_top100_{date}.txt"
    async with aiofiles.open(filename, 'w') as f:
        for line in data:
            await f.write(line + '\n')
    print(f"Arquivo {filename} gerado com sucesso!")

# Main function to process data in parallel
def process_data_in_parallel(data, num_workers=4):
    # Dividing the dataframe into chunks
    chunk_size = len(data) // num_workers
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Concatenating the results of all chunks
    return pd.concat(results)

# Function to read data from log file
# def read_data_from_file(file_path):
#     data = pd.read_csv(file_path, header=None, names=['player_id', 'match_id', 'operator_id', 'nb_kills'])
#     return data

# def read_data_from_file(file_path):
#     # Specify data types for better performance
#     dtype = {
#         'player_id': 'string',
#         'match_id': 'string',
#         'operator_id': 'Int32',  # Int32 for nullable integer
#         'nb_kills': 'Int8',
#     }
    
#     # Use iterator to read data in chunks
#     data_chunks = pd.read_csv(file_path, header=None, names=['player_id', 'match_id', 'operator_id', 'nb_kills'],
#                                dtype=dtype, chunksize=100000)  # Read 1 million rows at a time

#     # Concatenate all chunks into a single DataFrame
#     data = pd.concat(data_chunks, ignore_index=True)
    
#     return data

def read_data_from_file(file_path):
    # Specify data types for better performance
    dtype = {
        'player_id': 'string',
        'match_id': 'string',
        'operator_id': 'Int32',  # Int32 for nullable integer
        'nb_kills': 'Int8',
    }
    
    # Using iterator to read data in chunks and process each chunk immediately
    data_chunks = pd.read_csv(file_path, header=None, names=['player_id', 'match_id', 'operator_id', 'nb_kills'],
                               dtype=dtype, chunksize=100000)  # Read 100,000 rows at a time

    processed_results = []
    for chunk in data_chunks:
        processed_chunk = process_chunk(chunk)
        if not processed_chunk.empty:
            processed_results.append(processed_chunk)

    # Concatenate results of processed chunks
    return pd.concat(processed_results, ignore_index=True) if processed_results else pd.DataFrame()


# Function to monitor memory usage in real time
def monitor_memory_usage():
    process = psutil.Process()
    while True:
        mem_info = process.memory_info()
        print(f"Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB", end='\r')
        time.sleep(1)

# Main function to orchestrate processing
async def main():
    # Checks if a file was passed as an argument
    if len(sys.argv) != 2:  # Expects the first argument to be the file name
        print("Arquivo inválido ou faltando")
        sys.exit(1)

    file_path = sys.argv[1]
    data = read_data_from_file(file_path)

    # Start memory monitor in a separate thread
    monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    monitor_thread.start()

    # Process data in parallel
    processed_data = process_data_in_parallel(data, num_workers=4)
    
    # Calculate the top 100 operators
    top_100_data = calculate_operator_top100(processed_data)

    # Saving
    today = pd.Timestamp.now().strftime('%Y%m%d')
    await save_operator_top100(top_100_data, today)

if __name__ == "__main__":
    asyncio.run(main())
