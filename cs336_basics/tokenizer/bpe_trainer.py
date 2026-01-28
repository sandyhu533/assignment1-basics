import regex as re
import os
from typing import BinaryIO
from datetime import datetime
from multiprocessing import Process, Queue
import pickle
from cs336_basics.tokenizer.linkedlist import Node
import argparse


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't start on last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def do_pretoken(input_path, start, end, q, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")      
        pretokens = {}
        
        # Pre-tokenization
        ## 1. Split on special tokens
        special_pattern = "|".join(re.escape(t) for t in special_tokens)
        if special_pattern:
            chunks = re.split(special_pattern, chunk)
        else:
            chunks = [chunk]
        
        ## 2. Split on PAT
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            for m in re.finditer(PAT, chunk):
                str = m.group(0)
                encode_bytes = str.encode("utf-8")
                if len(encode_bytes) < 2:
                    continue
                val = pretokens.get(encode_bytes, 0)
                pretokens[encode_bytes] = val + 1
                
        # Put result into Queue
        q.put(pretokens)



def load_data(input_path: str | os.PathLike):
    path_obj = os.Path(input_path)
    
def update_pair_freq(pair_freq, pair, freq):
    pair_freq[pair] = pair_freq.get(pair, 0) + freq
    if pair_freq.get(pair, 0) == 0:
        pair_freq.pop(pair)
    
def update_pair_freq_and_loc(pair_freq, pair_loc, pair, node):
    update_pair_freq(pair_freq, pair, node.freq)
    locations = pair_loc.get(pair, [])
    locations.append(node)
    pair_loc[pair] = locations
                
class BPETrainer:
    def __init__(self):
        self.token2byte = {}
        self.byte2token = {}
        self.merges = []
        
    def new_token(self, val):
        id = len(self.token2byte)
        self.token2byte[id] = bytes(val)
        self.byte2token[bytes(val)] = id
    
    
    def train(self, input_path, vocab_size, special_tokens):
        # Init vocabs
        self.token2byte = {i: bytes([i]) for i in range(256)}
        self.byte2token = {bytes([i]): i for i in range(256)}
        for pretoken in special_tokens:
            token_value = pretoken.encode('utf-8')
            if self.byte2token.get(token_value) is None:
                self.new_token(token_value)
        
        # Load data into pretokens
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] ------start loading data and pretokenize----------")
        pretokens = {}
        q = Queue()
        processes = []
        with open(input_path, "rb") as f:
            num_processes = 10
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                p = Process(target=do_pretoken, args=(input_path, start, end, q, special_tokens))
                processes.append(p)
                p.start()
            
            for i in range(len(boundaries)-1):
                sub_pretokens=q.get()
                for k, v in sub_pretokens.items():
                    pretokens[k] = pretokens.get(k, 0) + v
            
            for p in processes:
                p.join()
        
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] ------initializing pairs----------")
        # Process pretokens
        pair_freq = {}
        pair_loc = {}
        for k, v in pretokens.items():
            # Process into linkedlist
            dummy = Node('',0)
            pre = dummy
            for b in k:
                node = Node(bytes([b]), v, pre)
                pre.next = node
                pre = node
            head = dummy.next
            head.pre = None
            
            node = head
            while(node.next != None):
                pair = node.next_pair()
                update_pair_freq_and_loc(pair_freq, pair_loc, pair, node)
                node = node.next
        
        
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] ------start merging tokens----------")
        step = 0
        while len(self.token2byte) < vocab_size and len(pair_freq) > 0:
            # Find the largest frequency pair and create new token
            best_key = max(pair_freq, key=lambda p: (pair_freq[p], p))
            max_freq = pair_freq.pop(best_key)
            self.merges.append(best_key)
            new_token = b"".join(best_key)
            self.new_token(new_token)
            newset_token = new_token
            
            if step % 100 == 0:
                print(f"best_tuple of step {step} is {best_key}, new_token is {new_token}, max_freq is {max_freq}")
            step += 1
            
            locations = pair_loc.pop(best_key)
            for node in locations:
                # Check if node location matches the best pair
                next_node = node.next
                if node == None or next_node == None or (node.token_bytes, next_node.token_bytes) != best_key:
                    continue
                
                # A, (B, C), D -> remove (C, D) and node C
                if next_node.next != None:
                    update_pair_freq(pair_freq, next_node.next_pair(), -next_node.freq)
                
                # A, (B, C), D -> remove (A, B)
                if node.pre != None:
                    update_pair_freq(pair_freq, node.pre_pair(), -node.pre.freq)
                
                # A, (B, C), D -> add (A, BC), (BC, D), change B to BC
                node.token_bytes = new_token
                next_node.token_bytes = b""
                node.merge()
                
                # print(f'new_token: {new_token} pre_pair: {node.pre_pair()} next_pair: {node.next_pair()}')
                if node.pre != None:
                    update_pair_freq_and_loc(pair_freq, pair_loc, node.pre_pair(), node.pre)
                if node.next != None:
                    update_pair_freq_and_loc(pair_freq, pair_loc, node.next_pair(), node)

        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] ------finish merging tokens----------")
        
        return self.token2byte, self.merges
    

import json

def save_tokenizer_json(vocab, merges, filename):
    # 将 bytes 转换为可序列化的格式 (这里用 list of ints)
    serializable_vocab = {
        token_id: str(token_bytes) 
        for token_id, token_bytes in vocab.items()
    }
    
    # 将 merges (tuple of bytes) 转换为 (list of ints)
    serializable_merges = [
        [str(p0), str(p1)] for p0, p1 in merges
    ]
    
    data = {
        "vocab": serializable_vocab,
        "merges": serializable_merges
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_tokenizer_pickle(vocab, merges, filename):
    data = {
        "vocab": vocab,
        "merges": merges
    }
    with open(filename, "wb") as f:
        # protocol=5 是目前最高性能的版本
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer on a text dataset.")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="TinyStoriesV2-GPT4-train.txt",
        help="Path to the training text file"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=10000,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/_model/",
        help="Directory to save the resulting tokenizer files"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Base name for output files (default: input filename). Use 'model' for model.pkl/model.json",
    )
    
    args = parser.parse_args()

    bpe = BPETrainer()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = args.output_name if args.output_name else os.path.basename(args.dataset)
    stem = base_name if args.output_name else f"{base_name}.result"
    save_path = os.path.join(args.output_dir, stem)
    
    print(f"Training on {args.dataset} with vocab size {args.vocab_size}...")
    
    token2byte, merges = bpe.train(
        args.dataset, 
        vocab_size=args.vocab_size, 
        special_tokens=["<|endoftext|>"]
    )
    
    save_tokenizer_json(token2byte, merges, save_path + ".json")
    save_tokenizer_pickle(token2byte, merges, save_path + ".pkl")
    print(f"Successfully saved to {save_path}.json and {save_path}.pkl")

if __name__ == '__main__':
    main()
