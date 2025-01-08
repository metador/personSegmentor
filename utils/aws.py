# ... existing imports ...
import boto3
from botocore.exceptions import ClientError
import os
# Add near the top of the file
from dotenv import load_dotenv

# Near the top of your file, after imports
# Load .env file if it exists
load_dotenv()

def init_s3_client():
    """Initialize S3 client with credentials"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_S3_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_S3_SECRET_KEY'),
        region_name=os.getenv('AWS_S3_REGION','us-east-1')  # e.g., 'us-east-1'
    )

def upload_to_s3(file_data, bucket_name, s3_path):
    """Upload a file to S3
    
    Args:
        file_data: The binary data to upload
        bucket_name: S3 bucket name
        s3_path: The path/name for the file in S3
    
    Returns:
        bool: True if file was uploaded, else False
    """
    s3_client = init_s3_client()
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_path,
            Body=file_data
        )
        return True
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return False

