import json
import pickle
import regex as re
import random
import time
from tqdm import tqdm
import numpy as np
import argparse
import os

def load_tokenizer_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["vocab"], data["merges"]

class BPETokenizer:
    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            token_bytes  = token.encode('utf-8')
            if token_bytes not in self.byte2token:
                id = len(self.vocab)
                self.vocab[id] = token
                self.byte2token[token] = id
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.byte2token = {v: k for k, v in self.vocab.items()}
        self.merge_order = {merge: i for i, merge in enumerate(merges)}
        if special_tokens:
            self.add_special_tokens(special_tokens)
        self.special_tokens = special_tokens or []
    
    @classmethod
    def from_files(cls, file_path, special_tokens=None):
        vocab, merges = load_tokenizer_pickle(file_path)
        return cls(vocab, merges, special_tokens)
    
    def get_avg_token_length(self):
       tokens = self.vocab.keys()
       total_length = sum(len(t) for t in tokens)
       return total_length / len(tokens)
   
    def get_compression_ratio(self, text):
        ids = self.encode(text)
        if not ids:
            return 0.0
        
        return len(text.encode("utf-8")) / len(ids)
    
    def get_longest_tokens(self, topk):
        sorted_items = sorted(self.vocab.values(), key=len, reverse=True)
        return sorted_items[:topk]
    
    def _bpe_merges(self, text):
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        
        while len(tokens)>=2:
            best_pair = min(
                zip(tokens[:-1], tokens[1:]),
                key=lambda p: self.merge_order.get(p, float('inf'))
            )
            if best_pair not in self.merge_order:
                break
            new_token = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == best_pair:
                    new_token.append(b''.join(best_pair))
                    i += 2
                else:
                    new_token.append(tokens[i])
                    i += 1
            tokens = new_token
        
        return tokens
        
        
    def encode(self, text):
        token_ids = []
        pretokens = []
        chunks = []

        # Pre-tokenization
        ## 1. Split on special tokens
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")"
        if special_pattern and special_pattern != "()":
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]
        
        ## 2. Split on PAT
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            if chunk is None:
                continue
            if chunk in self.special_tokens:
                pretokens.append(chunk)
            else:
                pretokens.extend(re.findall(PAT, chunk))
        
        # print(f'pretokens: {pretokens}')
        
        # Creat Linkedlist
        for token_str in pretokens:
            if token_str in self.special_tokens:
                tid = self.byte2token.get(token_str.encode('utf-8'))
                if tid is not None: token_ids.append(tid)
                continue
            
            token_bytes = self._bpe_merges(token_str)
            # Convert to token_ids
            for token_byte in token_bytes:
                if token_byte in self.byte2token:
                    token_ids.append(self.byte2token[token_byte])
        
        return token_ids
    
    def encode_iterable(self, iterable):
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids
    
    def decode(self, ids):
        text = b''
        for id in ids:
            if id in self.vocab:
                text += self.vocab.get(id)
        return text.decode(errors="replace")
    

def diff(tokenizer1, tokenizer2):
    # Compare diff between two tokenizer
    vocab1_set = set(tokenizer1.vocab.values())
    vocab2_set = set(tokenizer2.vocab.values())

    shared_tokens = vocab1_set.intersection(vocab2_set)
    shared_count = len(shared_tokens)

    overlap_ratio = shared_count / len(vocab1_set)

    print(f"vocab1_size: {len(vocab1_set)} vocab2_size: {len(vocab2_set)}")
    print(f"Shared tokens: {shared_count}")
    print(f"Overlap ratio: {overlap_ratio:.2%}")
    
def get_chunks(input_path, special_tokens=["<|endoftext|>"]):
    chunks = []
    with open(input_path, "rb") as f:
        chunk = f.read().decode("utf-8", errors="ignore")      
        pretokens = {}
    
        # Pre-tokenization
        ## 1. Split on special tokens
        special_pattern = "|".join(re.escape(t) for t in special_tokens)
        if special_pattern:
            chunks = re.split(special_pattern, chunk)
        else:
            chunks = [chunk] 
    return chunks

def test():
    tiny_stories_bpe = "data/_model/TinyStoriesV2-GPT4-train.txt"
    # dataset = "tests/_model/tinystories_sample_5M.txt"
    
    ts_tokenizer = BPETokenizer.from_files(tiny_stories_bpe+".result.pkl", special_tokens=["<|endoftext|>"])
    str_list = [b.decode('utf-8') for b in ts_tokenizer.get_longest_tokens(10)]
    print(str_list)
    strlist = ["1", " accomplishment", "the cat ate", "🙃"]
    for st in strlist:
        print(f'original: {st} encode: {ts_tokenizer.encode(st)} decode: {ts_tokenizer.decode(ts_tokenizer.encode(st))}')
    
    owt_bpe = "data/_model/owt_train.txt"
    owt_tokenizer = BPETokenizer.from_files(owt_bpe+".result.pkl", special_tokens=["<|endoftext|>"])
    str_list = [b.decode('utf-8') for b in owt_tokenizer.get_longest_tokens(10)]
    print(str_list)
    strlist = ["1", " accomplishment", "the cat ate", "🙃"]
    for st in strlist:
        print(f'original: {st} encode: {owt_tokenizer.encode(st)} decode: {owt_tokenizer.decode(owt_tokenizer.encode(st))}')
    
    diff(ts_tokenizer, owt_tokenizer)
    
    tiny_stories_valid = "data/TinyStoriesV2-GPT4-valid.txt"
    owt_valid = "data/owt_valid.txt"
    
    tiny_stories_sample = random.sample(get_chunks(tiny_stories_valid), 10)
    owt_sample = random.sample(get_chunks(owt_valid), 10)
    
    tsts = sum(ts_tokenizer.get_compression_ratio(chunk) for chunk in tiny_stories_sample) / len(tiny_stories_sample)
    print(f'compression rate of tiny_stories_sample with ts_tokenizer={tsts}')
    
    tsowt = sum(owt_tokenizer.get_compression_ratio(chunk) for chunk in tiny_stories_sample) / len(tiny_stories_sample)
    print(f'compression rate of tiny_stories_sample with owt_tokenizer={tsowt}')
    
    owtts = sum(ts_tokenizer.get_compression_ratio(chunk) for chunk in owt_sample) / len(owt_sample)
    print(f'compression rate of owt_sample with ts_tokenizer={owtts}')
    
    owtowt = sum(owt_tokenizer.get_compression_ratio(chunk) for chunk in owt_sample) / len(owt_sample)
    print(f'compression rate of owt_sample with owt_tokenizer={owtowt}')
    
    start_time = time.perf_counter()
    text = "<|endoftext|>".join(owt_sample)
    nbytes = len(text)
    owt_tokenizer.encode(text)
    end_time = time.perf_counter()
    execution_time = end_time-start_time
    print(f'estimated bytes/s of owt_tokenizer is {nbytes/execution_time}')
    
    start_time = time.perf_counter()
    text = "<|endoftext|>".join(tiny_stories_sample)
    nbytes = len(text)
    ts_tokenizer.encode(text)
    end_time = time.perf_counter()
    execution_time = end_time-start_time
    print(f'estimated bytes/s of ts_tokenizer is {nbytes/execution_time}')

def serialize_dataset(tokenizer, input_file, output_file):
    token_ids = []
    
    print(f"Encoding {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if line.strip():
                ids = tokenizer.encode(line)
                token_ids.extend(ids)
    
    ids_array = np.array(token_ids, dtype=np.uint16)
    
    np.save(output_file, ids_array)
    print(f"Serialized to {output_file}. Final shape: {ids_array.shape}")
    
def main():
    parser = argparse.ArgumentParser(description="Serialize text dataset to NumPy uint16 array.")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Path to .result.pkl or .json tokenizer file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw .txt file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save .npy file")
    
    args = parser.parse_args()

    # 1. 加载 Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_model}...")
    tokenizer = BPETokenizer.from_files(args.tokenizer_model)

    # 2. 准备输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 3. 逐行读取并编码
    token_ids = []
    print(f"Encoding {args.input_file}...")
    
    num_lines = sum(1 for _ in open(args.input_file, 'r', encoding='utf-8'))
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            # 使用 tqdm 显示处理进度
            for line in tqdm(f, total=num_lines, desc="Processing"):
                if line.strip():
                    # 调用你实现的 encode 方法
                    ids = tokenizer.encode(line)
                    token_ids.extend(ids)
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found.")
        return

    # 4. 序列化为 uint16 NumPy 数组
    print("Converting to NumPy array (uint16)...")
    ids_array = np.array(token_ids, dtype=np.uint16)

    # 5. 保存
    np.save(args.output_file, ids_array)
    print(f"Successfully saved {len(ids_array)} tokens to {args.output_file}")
    print(f"File size: {os.path.getsize(args.output_file) / (1024**2):.2f} MB")

if __name__ == '__main__':
    # test()
    main()