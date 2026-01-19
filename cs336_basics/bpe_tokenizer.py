import json
import pickle
import regex as re
from .linkedlist import Node

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
    
    def get_longest_tokens(self, topk):
        sorted_items = sorted(self.vocab.items(), key=lambda x: len(x[1]), reverse=True)
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
    

if __name__ == '__main__':
    dataset = "TinyStoriesV2-GPT4-train.txt"
    # dataset = "tests/fixtures/tinystories_sample_5M.txt"
    # dataset = "owt_train.txt"
    tokenizer = BPETokenizer.from_files(dataset+".result.pkl", special_tokens=["<|endoftext|>"])
    print(tokenizer.get_longest_tokens(10))
    strlist = ["1", " accomplishment", "the cat ate", "🙃"]
    for st in strlist:
        print(f'original: {st} encode: {tokenizer.encode(st)} decode: {tokenizer.decode(tokenizer.encode(st))}')
