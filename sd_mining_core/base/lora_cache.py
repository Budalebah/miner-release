from collections import OrderedDict

class LoRACache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.unique_ids = {}

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                lru_key, _ = self.cache.popitem(last=False)
                del self.unique_ids[lru_key]
                return lru_key
            if key not in self.unique_ids:
                self.unique_ids[key] = f"default_{len(self.unique_ids)}"
        self.cache[key] = value
        return None

    def get_unique_id(self, key):
        return self.unique_ids.get(key)

    def remove(self, key):
        if key in self.cache:
            del self.cache[key]
            del self.unique_ids[key]