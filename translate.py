# Google Cloud SDK Translate
from deep_translator.exceptions import RequestError
from google.cloud.translate_v3 import TranslateTextResponse

import config
from google.cloud import translate
from deep_translator import GoogleTranslator


def get_cloud_configs():
    config.set_google_credentials()
    parent = f"projects/{config.PROJECT_ID}"
    client = translate.TranslationServiceClient()
    return parent, client


def google_cloud_translate(content: list[str], parent: str,
                           client: translate.TranslationServiceClient) -> TranslateTextResponse:
    return client.translate_text(
        request={
            "parent": parent,
            "contents": content,
            "mime_type": "text/plain",
            "source_language_code": "en-US",
            "target_language_code": "si",
        }
    )


def deep_translate(text: str) -> str:
    return GoogleTranslator(source='en', target='si').translate(text=text)
