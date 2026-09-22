import pandas as pd
from pymongo import MongoClient

# Local MongoDB साठी connection URI
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "student_db"
COLLECTION_NAME = "student_records"

def upload_csv_to_mongodb(csv_path):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    
    collection.delete_many({})  # जुना डेटा साफ करण्यासाठी
    collection.insert_many(records)
    print(f"✅ Successfully uploaded {len(records)} records to MongoDB!")
    client.close()

if __name__ == "__main__":
    upload_csv_to_mongodb("data/student_data.csv")