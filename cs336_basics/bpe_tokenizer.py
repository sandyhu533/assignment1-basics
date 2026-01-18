import regex as re
import os
from typing import BinaryIO
from datetime import datetime
from multiprocessing import Process, Queue


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
    # Chunks start on previous index, don't include last index
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
            # print(chunk)
            # print(re.findall(PAT, chunk))
            for m in re.finditer(PAT, chunk):
                str = m.group(0)
                encode_bytes = str.encode("utf-8")
                byte_tuple = tuple(bytes([b]) for b in encode_bytes)
                if len(byte_tuple) < 2:
                    continue
                val = pretokens.get(byte_tuple, 0)
                pretokens[byte_tuple] = val + 1
                
        # Put result into Queue
        q.put(pretokens)



def load_data(input_path: str | os.PathLike):
    path_obj = os.Path(input_path)
    
class BPETokenizer:
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
        for token in special_tokens:
            token_value = token.encode('utf-8')
            if self.byte2token.get(token_value) is None:
                self.new_token(token_value)
        
        # Load data
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
        print(f"[{now}] ------start merging tokens----------")
        step = 0
        newset_token = bytes()
        # Merge token util hits vocab_size
        while len(self.token2byte) < vocab_size and len(pretokens) != 0:
            step += 1
            if step % 100 == 0:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] current step: {step} newest_token: {newset_token}")
            pair_freq = {}
            pair_origin_word_list = {}
            # Count the freq of each pair
            for token, token_freq in pretokens.items():
                for i in range(len(token)-1):
                    byte_pair = (token[i],token[i+1])
                    pair_freq_val = pair_freq.get(byte_pair, 0)
                    pair_freq[byte_pair] = pair_freq_val + token_freq
                    if byte_pair not in pair_origin_word_list:
                        pair_origin_word_list[byte_pair] = set()
                    pair_origin_word_list[byte_pair].add(token)
            if len(pair_freq) == 0:
                break
            
            # Assign the max freq pair as new token
            best_key = max(pair_freq, key=lambda p: (pair_freq[p], p))
            self.merges.append(best_key)
            new_token = b"".join(best_key)
            self.new_token(new_token)
            newset_token = new_token
            # pre_token_length = len(pretokens)
            # print(f"best_tuple of step {step} is {best_key}, new_token is {new_token}, pre_token_length is {pre_token_length}")
            # print(pair_origin_word_list.get(best_key, []))
            
            # Replace the origin pretokens
            for word in pair_origin_word_list.get(best_key, []):
                freq = pretokens.pop(word, 0)
                # print(f"removed {word}")
                new_pre_token = ()
                i = 0
                while i < len(word):
                    if i != len(word) - 1 and best_key == (word[i], word[i+1]):
                        new_pre_token = (*new_pre_token, new_token)
                        i += 1
                    else:
                        new_pre_token = (*new_pre_token, word[i])
                    i += 1
                if len(new_pre_token) > 1:
                    pretokens[new_pre_token] = pretokens.get(new_pre_token, 0) + freq
                    # print(f"added {new_pre_token}")

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

if __name__ == '__main__':
    bpe = BPETokenizer()
    dataset = "TinyStoriesV2-GPT4-train.txt"
    # dataset = "tests/fixtures/tinystories_sample_5M.txt"
    resultPath = dataset+".result.json"
    token2byte, merges = bpe.train(dataset, vocab_size=10000, special_tokens=["<|endoftext|>"])
    save_tokenizer_json(token2byte, merges, resultPath)