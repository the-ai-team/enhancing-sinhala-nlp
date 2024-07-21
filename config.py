from os import environ
import json

GOOGLE_APPLICATION_CREDENTIALS = './keys/service-account.json'


def set_google_credentials():
    environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS


def get_project_id():
    with open(GOOGLE_APPLICATION_CREDENTIALS) as f:
        credentials = json.load(f)
    return credentials.get('project_id')


PROJECT_ID = get_project_id()
# PROJECT_ID = '123'

SPREADSHEET_NAME = 'sample_data.xlsx'


def get_file_name(folder: str, index: int, suffix: str, ext: str = 'csv') -> str:
    return f'{folder}/{index}-{suffix}.{ext}'
