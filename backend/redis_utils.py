import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()
redis_url = os.getenv("REDIS_URL")

redis_client = redis.Redis.from_url(url=redis_url, decode_responses=True)

def get_chat_key(session_id: str):
    return f"chat:{session_id}"

def load_chat_history(session_id: str):
    key = get_chat_key(session_id)
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return []

def save_chat_history(session_id: str, history: list):
    key = get_chat_key(session_id)
    redis_client.set(key, json.dumps(history), ex=3600 * 6) 
