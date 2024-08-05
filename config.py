from os import environ
import json

GOOGLE_APPLICATION_CREDENTIALS = './keys/service-account.json'
environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

def get_project_id():
    with open(GOOGLE_APPLICATION_CREDENTIALS) as f:
        credentials = json.load(f)
    return credentials.get('project_id')
