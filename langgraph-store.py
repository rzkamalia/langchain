
import uuid

from langgraph.store.memory import InMemoryStore


in_memory_store = InMemoryStore()

# when storing memories in the Store, we provide: namespace, key, value
user_id = "1"
namespace = ("user_id", "memories")

key = str(uuid.uuid4())

value = {"food_preference" : "I like pizza"}

in_memory_store.put(namespace, key, value)

# to get the memory by namespace and key
memory = in_memory_store.get(namespace, key)
print(memory.dict())