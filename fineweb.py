import os
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb100M"
remote_name = "sample-10BT"
max_tokens = int(1e8)  # Tokens per shard; adjust as needed
max_shards = 2       # Maximum number of shards to produce

# Create the cache directory if it doesn't exist yet.
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset in streaming mode.
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

# Initialize the tokenizer.
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end-of-text token

def tokenize(doc):
    """
    Tokenizes a single document and returns a numpy array of uint16 tokens.
    The special <|endoftext|> token is prepended to delimit documents.
    """
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # Ensure tokens fit into uint16.
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# --- Sharded token accumulation ---
current_shard_tokens = []  # list to hold numpy arrays of tokens
current_token_count = 0    # token count in current shard
shard_idx = 0              # index to count shards

progress_bar = tqdm(unit="tokens", desc="Accumulating tokens")

# Process the streamed dataset.
for doc in fw:
    tokens = tokenize(doc)
    # If adding tokens does not exceed the shard limit:
    if current_token_count + len(tokens) < max_tokens:
        current_shard_tokens.append(tokens)
        current_token_count += len(tokens)
        progress_bar.update(len(tokens))
    else:
        # If not enough room, fill current shard with as many tokens as possible.
        remaining = max_tokens - current_token_count
        if remaining > 0:
            current_shard_tokens.append(tokens[:remaining])
            current_token_count += remaining
            progress_bar.update(remaining)
        # Write the current shard to file.
        all_tokens_np = np.concatenate(current_shard_tokens)
        output_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_shard_{shard_idx:03d}_tokens.npy")
        write_datafile(output_filename, all_tokens_np)
        print(f"Finished! Wrote {current_token_count} tokens to {output_filename}")
        shard_idx += 1
        # If we've reached the desired number of shards, stop processing.
        if shard_idx >= max_shards:
            break
        # Prepare for the next shard with any leftover tokens from the current doc.
        leftover = tokens[remaining:]
        current_shard_tokens = []
        current_token_count = 0
        if leftover.size > 0:
            current_shard_tokens.append(leftover)
            current_token_count += len(leftover)
            progress_bar.update(len(leftover))

progress_bar.close()

# If loop finishes without reaching max_shards and some tokens remain, write them to a final shard.
if shard_idx < max_shards and current_token_count > 0:
    all_tokens_np = np.concatenate(current_shard_tokens)
    output_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_shard_{shard_idx:03d}_tokens.npy")
    write_datafile(output_filename, all_tokens_np)
    print(f"Finished! Wrote {current_token_count} tokens to {output_filename}")

import os
os._exit(0)