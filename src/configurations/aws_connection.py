import boto3
import sys
import os
from dotenv import load_dotenv
from src.exception import CustomException


class AWSClient:
    client = None
    
    def __init__(self):
        try:
            load_dotenv()
            if AWSClient.client is None:
                self.initialise_aws_client()
        except Exception as e:
            raise CustomException(e, sys)
        
    def initialise_aws_client(self):
        try:
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_REGION')

            AWSClient.client = boto3.client(
                's3',
                aws_access_key_id = aws_access_key_id,
                aws_secret_access_key = aws_secret_access_key,
                region_name = aws_region
            )

        except Exception as e:
            raise CustomException(e, sys)