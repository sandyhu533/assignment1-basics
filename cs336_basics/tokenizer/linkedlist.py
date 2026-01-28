class Node:
    def __init__(self, token_bytes, freq=0, pre=None):
        self.token_bytes = token_bytes
        self.freq = freq
        self.pre = pre
        self.next = None
        
    def merge(self):
        next = self.next
        if next == None:
            return
        new_next = next.next
        self.next = new_next
        if new_next != None:
           new_next.pre = self
    
    def pre_pair(self):
        if self.pre == None:
            return ()
        return (self.pre.token_bytes, self.token_bytes)
        
    def next_pair(self):
        if self.next == None:
            return ()
        return (self.token_bytes, self.next.token_bytes)
