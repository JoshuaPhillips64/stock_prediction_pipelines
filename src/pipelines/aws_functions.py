import boto3
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY

"""
S3
"""

def open_s3_resource_connection():
    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY, )

    s3 = session.resource('s3')
    return s3

def pull_from_s3(bucket_name, s3_file_name, local_file_name):
    """
    Downloads file from s3 bucket - ideal for downloading.
    If you want to use it as part of script -
        use object_contents = s3.Object('mybucket', 'myfile.txt').get()['Body'].read()
    Example usage:pull_from_s3('mybucket', 'myfile.txt', '/path/to/save/myfile.txt')

    :return:
    """
    s3 = open_s3_resource_connection()
    return s3.Bucket(bucket_name).download_file(s3_file_name, local_file_name)

def s3_upload_file(current_file_location:str,bucket_name:str,desired_file_name:str,folder_name: str = ''):
    """"
    Uploads file using connection parameters
    """
    s3 = open_s3_resource_connection()
    file_name_with_type = f'{desired_file_name}'

    # If folder_name is provided, prepend it to the file name with a forward slash
    if folder_name:
        s3_key = f'{folder_name}/{file_name_with_type}'
    else:
        s3_key = file_name_with_type

    try:
        s3.meta.client.upload_file(Filename=current_file_location, Bucket=bucket_name, Key=s3_key)
        print(f"Upload Successful of " + f'{desired_file_name}' )
    except FileNotFoundError:
        print("The file was not found")

def pull_from_s3_bucket_using_last_updated(target_bucket:str,minutes_since_updated:int):
    """
    Loops through s3 bucket to find the file that has been updated in last X minutes. For this example time is set to 15 minutes.

    :param target_bucket:
    :return: dataframe
    """
    s3 = open_s3_resource_connection()

    # set the delta to be 15 min before
    delta = timedelta(
        days=0,
        seconds=0,
        microseconds=0,
        milliseconds=0,
        minutes=minutes_since_updated,
        hours=0,
        weeks=0)

    bucketreference = s3.Bucket(f'{target_bucket}')
    # %%Loop through s3 bucket
    for file in bucketreference.objects.all():
        # compare dates and only pull last day based on delta above
        if (file.last_modified).replace(tzinfo=None) > (datetime.now() - delta):
            # print results
            newfile = file.key
            print('File Name: %s ---- Date: %s' % (file.key, file.last_modified))

    bucket_object = s3.Object(
        bucket_name=f'{target_bucket}',
        key=file.key
    ).get()

    object_contents = bucket_object['Body'].read()

    return object_contents

def pull_from_s3_bucket_using_text(s3_connection, target_bucket:str,search_text:str,folder:str = '',suffix: str = ''):
    """
    Updated to require external persistent s3 connections to perform better in loops.
    Function Loops through s3 bucket to find the files that contain the terms in search text.

    :param target_bucket:
    :return: dataframe
    """

    bucket = s3_connection.Bucket(target_bucket)
    prefix = folder + '/' if folder else ''
    object_contents = None

    for file in bucket.objects.filter(Prefix=prefix):
        if search_text in file.key and (not suffix or file.key.endswith(suffix)):
            print(f'File Name: {file.key} ---- Last Updated Date: {file.last_modified}')
            object_contents = file.get()['Body'].read()
            break

    if object_contents:
        return object_contents
    else:
        print(f'Unable to locate file with {search_text} in {target_bucket}')
        return None


def loop_through_s3_folder(s3_resource, bucket_name, folder_to_search, text_to_search_for,
                           additional_text_to_search_for=""):
    """
    Loop through S3 folder and filter for files containing specific text(s), collecting file keys. Does not download files.

    :param s3_resource: Active S3 resource connection.
    :param bucket_name: Name of the S3 bucket.
    :param folder_to_search: Folder prefix in S3 to search.
    :param text_to_search_for: List of texts to search for in filenames.
    :param additional_text_to_search_for: Additional text to search for in filenames. Default is "".
    :return: List of file keys that match the criteria.
    """
    file_keys = []
    try:
        bucket = s3_resource.Bucket(bucket_name)
        print('Connection Successful')

        for obj in bucket.objects.filter(Prefix=folder_to_search):
            if any(text in obj.key for text in text_to_search_for) and additional_text_to_search_for in obj.key:
                file_keys.append(obj.key)
                print(f'File Name: {obj.key} appended to list to download')

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print('Successfully completed')
    return file_keys

"""
Simple Notification Service
"""

def send_sns_email(failed_job_name:str,error_message:str,SNS_arn,region_name):
    """
    If an error will send to SNS Topic.
    Need to ensure SNS topic is set up correctly. https://docs.aws.amazon.com/sns/latest/dg/sns-getting-started.html#step-create-queue
    :param function_to_run:
    :return:
    """
    sns_client = boto3.client('sns',region_name=region_name)
    print(f'Error when running function {failed_job_name} Error code: {error_message} ')
    sns_client.publish(TopicArn=SNS_arn,
                       Message=f'Error when running function {failed_job_name},Error code: {error_message}',
                       Subject=f'Error when running function {failed_job_name}')