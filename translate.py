# Google Cloud SDK Translate
from typing import List
from deep_translator.exceptions import RequestError
from google.cloud.translate_v3 import TranslateTextResponse

import config
from google.cloud import translate
from deep_translator import GoogleTranslator


def get_cloud_configs():
    project_id = config.get_project_id()
    parent = f"projects/{project_id}"
    client = translate.TranslationServiceClient()
    return parent, client


def google_cloud_translate(content: list[str]) -> List[str]:
    parent, client = get_cloud_configs()
    result = client.translate_text(
        request={
            "parent": parent,
            "contents": content,
            "mime_type": "text/plain",
            "source_language_code": "en-US",
            "target_language_code": "si",
        }
    )
    return [translation.translated_text for translation in result.translations]


def deep_translate(text: str) -> str:
    return GoogleTranslator(source='en', target='si').translate(text=text)
