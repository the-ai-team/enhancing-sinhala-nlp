import re
import time
import os
from typing import List, Any
import pandas as pd
from deep_translator.exceptions import BaseError, RequestError
from pandas import DataFrame
from termcolor import colored

from errors import CannotSplitIntoChunksError, EmptyContentError, MaxChunkSizeExceededError, \
    DelimiterAlreadyExistsError, TranslateIOMismatchError, \
    DatasetParquetNameError, TranslationError
from multi_thread_handler import mth


def read_integer_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.readline().strip()
            if content:
                value = int(content)
                return value
            else:
                return 0
    except FileNotFoundError:
        mth.safe_print(f"File not found: {file_path}")
        return 0
    except ValueError:
        mth.safe_print(f"Invalid integer value in file: {file_path}")
        return 0


def update_integer_in_file(file_path, new_value):
    try:
        with open(file_path, 'w') as file:
            file.write(str(new_value))
    except Exception as e:
        mth.safe_print(f"Error updating file: {e}")


def get_current_time():
    return int(time.time())


def load_dataset(folder_path: str, start: int = None, end: int = None) -> DataFrame:
    all_files = os.listdir(folder_path)

    # Filter out only Parquet files
    parquet_files = [f for f in all_files if f.endswith('.parquet')]

    for file in parquet_files:
        match = re.search(r'part\.(\d+)\.parquet', file)
        if match is None:
            raise DatasetParquetNameError()

    sorted_parquet_files = sorted(parquet_files, key=lambda x: int(re.search(r'part\.(\d+)\.parquet', x).group(1)))

    dataframes = []

    # Read each Parquet file and append to the list
    for file in sorted_parquet_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_parquet(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes).reset_index(drop=True)

    # Apply slicing if start and/or end are not None
    if start is not None and end is not None:
        return combined_df.iloc[start:end]
    elif start is not None:
        return combined_df.iloc[start:]
    elif end is not None:
        return combined_df.iloc[:end]
    else:
        return combined_df


default_split_delimiters = ['\n', '.']

def split_text_into_chunks(text, split_delimiters=default_split_delimiters, chunk_size=4000) -> List[str]:
    def find_best_delimiter(text_slice):
        best_delimiter = None
        best_index = -1
        for delimiter in split_delimiters:
            index = text_slice.rfind(delimiter)
            if index > best_index:
                best_index = index
                best_delimiter = delimiter
        return best_delimiter, best_index
    
    chunks = []

    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break

        text_slice = text[:chunk_size]
        best_delimiter, split_index = find_best_delimiter(text_slice)

        if split_index == -1:
            raise CannotSplitIntoChunksError()

        index_with_delimiter = split_index + len(best_delimiter)
        chunks.append(text[:index_with_delimiter])
        text = text[index_with_delimiter:]

    return chunks


def connect_back_chunks(chunks: List[str]) -> str:
    return ''.join(chunks)


def escape_delimiters(text: str) -> str:
    return repr(text)[1:-1]


def translate_by_chunk(translate_fn: callable, text: str, chunk_size=4000, split_delimiters=default_split_delimiters) -> str:
    if len(text) <= chunk_size:
        return translate_fn(text)

    chunks = split_text_into_chunks(text, split_delimiters, chunk_size)

    translated_chunks = [translate_fn(chunk) for chunk in chunks]
    translated_content = connect_back_chunks(translated_chunks)
    mth.safe_print(
        f"Translated {len(chunks)} chunks after splitting.")

    return translated_content


combine_delimiters = ['\n<###>\n']


def combine_text_into_blob(content: List[str], combine_delimiter=combine_delimiters[0], max_chunk_size=4000) -> str:
    # Combine text
    combined_text = ""
    for i, text in enumerate(content):
        if i < len(content) - 1:
            combined_text += text + combine_delimiter
        else:
            combined_text += text

    if len(combined_text) >= max_chunk_size:
        raise MaxChunkSizeExceededError()
    return combined_text


def split_blob_into_text(blob: str, combine_delimiter=combine_delimiters[0]) -> List[str]:
    return blob.split(combine_delimiter)


def translate_by_blob(translate_fn: callable, content: List[str], max_chunk_size=4000) -> List[str]:
    # Check whether delimiter is already exists in the text
    for text in content:
        if not text.strip():
            raise EmptyContentError()
        if combine_delimiters[0] in text:
            raise DelimiterAlreadyExistsError()

    combined_text = combine_text_into_blob(content, combine_delimiters[0], max_chunk_size)
    translated_text = translate_fn(combined_text)
    split_content = split_blob_into_text(translated_text, combine_delimiters[0])

    if len(split_content) != len(content):
        raise TranslateIOMismatchError()

    return split_content


def translate_by_sdk(translate_sdk_fn: callable, content: List[str], max_chunk_size=30000) -> List[str]:
    # Check if the total content length exceeds the maximum chunk size. If it does not, translate the content as a whole
    total_content_length = sum([len(text) for text in content])
    if total_content_length <= max_chunk_size:
        return translate_sdk_fn(content)


    translated_content = []
    split_delimeters = ['\n', '.', '   ']
    for i, text in enumerate(content):
        if len(text) >= max_chunk_size:
            chunks = split_text_into_chunks(text, split_delimiters=split_delimeters, chunk_size=max_chunk_size)

            translated_chunks = []
            for chunk in chunks:
                translated_chunks.extend(translate_sdk_fn([chunk]))

            translated_text = connect_back_chunks(translated_chunks)
            translated_content.append(translated_text)

        else:
            translated_content.extend(translate_sdk_fn([text]))

    return translated_content



def choose_translation_method_and_translate(translate_fn: callable, translate_sdk_fn: callable, index: int, content: List[str],
                                            max_chunk_size=4000) -> List[str]:
    try:
        try:
            translated_content = translate_by_blob(translate_fn, content, max_chunk_size)

            current_time = get_current_time()
            mth.safe_print(f"Translated by blob for index {index}, Time: {current_time}")
            return translated_content
        
        except (TranslationError, RequestError) as e:
            try:
                mth.safe_print(colored(f"Translate by blob for index {index} failed, trying by chunks. Error: {e}", "yellow"))
                translated_content = [translate_by_chunk(translate_fn, text, max_chunk_size) for text in content]

                current_time = get_current_time()
                mth.safe_print(f"Translated by chunk for index {index}, Time: {current_time}")
                return translated_content
            except (TranslationError, RequestError) as e:
                mth.safe_print(colored(f"Translate by chunk for index {index} failed, trying with SDK. Error: {e}", "yellow"))
                translated_content = translate_by_sdk(translate_sdk_fn, content)

                current_time = get_current_time()
                mth.safe_print(f"Translated by SDK for index {index}, Time: {current_time}")
                return translated_content

    except BaseError as e:
        print(colored(f"Deep Translator Error: {e.message} at {index}", 'red'))
        raise e
    except Exception as e:
        print(colored(f"Error: {e} at {index}", 'red'))
        raise e


def get_estimated_time(content_len, i, start_time, current_time):
    if i == 0:
        return 0
    elapsed_time = current_time - start_time
    avg_time_per_chunk = elapsed_time / i
    remaining_chunks = content_len - i
    estimated_time = (avg_time_per_chunk * remaining_chunks) / 3600
    return round(estimated_time, 2)


def get_speed(content_len, start_time, current_time):
    elapsed_time = current_time - start_time
    if elapsed_time == 0:
        return 0
    speed = content_len / elapsed_time
    return round(speed, 2)


def get_output_csv_path(folder: str, index: int, suffix: str, ext: str = 'csv') -> str:
    return f'{folder}/{index}-{suffix}.{ext}'
