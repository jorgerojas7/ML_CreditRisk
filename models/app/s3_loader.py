import boto3
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


BUCKET_NAME = 'anyoneai-datasets'
PREFIX = 'credit-data-2010/'

DOWNLOAD_DIR = os.path.join('data', 'raw')

def download_data_from_s3():
    s3 = boto3.client(
        's3', 
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    files = []
    for obj in response.get('Contents', []):
        file_key = obj['Key']
        file_name = file_key.split('/')[-1]
        if file_name:
            dest_path = os.path.join(DOWNLOAD_DIR, file_name)
            print(f"📥 Downloading {file_name}")
            s3.download_file(BUCKET_NAME, file_key, dest_path)
            files.append(dest_path)
    return files
