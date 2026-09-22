import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "student_db"
COLLECTION_NAME = "student_records"

def load_data_from_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Fetch records excluding MongoDB internal `_id` field
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    client.close()
    return df