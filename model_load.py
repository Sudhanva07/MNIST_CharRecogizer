import boto3

s3 = boto3.client("s3")

response = s3.list_objects_v2(
    Bucket="mnist-model-storage-024757002421-ap-southeast-2-an"
)

for obj in response.get("Contents", []):
    print(obj["Key"])

s3.download_file(
    "mnist-model-storage-024757002421-ap-southeast-2-an",
    "model.pkl",
    "model.pkl"
)