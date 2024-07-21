### Translate using unofficial API (without multi-threading)
# import csv
# import config, utils
#
# folder_path = 'cot-fspot'
# dataset = utils.load_dataset(folder_path)
#
# start_pointer_file_path = 'outputs/start-pointer.txt'
# next_file_index_file_path = 'outputs/next-file-index.txt'
#
# start_pointer = utils.read_integer_from_file(start_pointer_file_path)
# next_file_index = utils.read_integer_from_file(next_file_index_file_path)
#
#
# def translate(text: str) -> str:
#     result = GoogleTranslator(source='en', target='si').translate(text=text)
#     return result
#
#
# file_name = config.get_file_name('outputs', next_file_index, 'cot-fspot', 'csv')
# start_time = utils.get_current_time()
#
# block_end = 120000
#
# with open(file_name, 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Id', 'Original Input', 'Translated Input', 'Original Target', 'Translated Target'])
#
#     utils.update_integer_in_file(next_file_index_file_path, next_file_index + 1)
#
#     content_len = len(dataset)
#
#     for i, row in dataset.loc[start_pointer:block_end].iterrows():
#         input_text = row['inputs']
#         target_text = row['targets']
#
#         input_result = utils.translate_by_chunk(translate, input_text)
#         target_result = utils.translate_by_chunk(translate, target_text)
#         # input_result = utils.connect_chunks([translate(chunk) for chunk in utils.split_text_into_chunks(input_text)])
#         # target_result = utils.connect_chunks([translate(chunk) for chunk in utils.split_text_into_chunks(target_text)])
#
#         writer.writerow([i, input_text, input_result, target_text, target_result])
#         utils.update_integer_in_file(start_pointer_file_path, i + 1)
#
#         current_time = utils.get_current_time()
#         speed = utils.get_speed(i - start_pointer, start_time, current_time)
#         estimated_time = utils.get_estimated_time(block_end - start_pointer, i - start_pointer, start_time,
#                                                   current_time)
#
#         # Prints count(c) starting from 1. (index = c - 1)
#         print(
#             f"Translated {i + 1} of {content_len}, Elapsed (Secs): {current_time - start_time}, Estimated (Hrs): {estimated_time}, Speed: {speed}")

### Translate using unofficial API (with multi-threading)
# from errors import InvalidOutputError
# import csv
# import concurrent.futures
# import config, utils
# from multi_thread_handler import MultiThreadHandler
#
# start_pointer_file_path = 'outputs-1k/start-pointer.txt'
# next_file_index_file_path = 'outputs-1k/next-file-index.txt'
#
# start_pointer = utils.read_integer_from_file(start_pointer_file_path)
# next_file_index = utils.read_integer_from_file(next_file_index_file_path)
#
# file_name = config.get_file_name('outputs-1k', next_file_index, 'cot-zspot', 'csv')
# start_time = utils.get_current_time()
# content_len = len(dataset)
# # start_pointer = 111400
# # next_file_index = 100
# # block_end = 150000
#
# mth = MultiThreadHandler()
#
#
# def process_row(args):
#     i, row = args
#     input_text = row['inputs']
#     target_text = row['targets']
#
#     result = utils.choose_translation_method_and_translate(mth.rate_limited_translate, i + 1, [input_text, target_text])
#     if len(result) != 2:
#         raise InvalidOutputError
#
#     input_result = result[0]
#     target_result = result[1]
#
#     mth.safe_print(f"Queued Translation: {i + 1}")
#     return i, input_text, input_result, target_text, target_result
#
#
# def translate_dataset(block_end: int = None):
#     with open(file_name, 'w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Id', 'Original Input', 'Translated Input', 'Original Target', 'Translated Target'])
#
#         utils.update_integer_in_file(next_file_index_file_path, next_file_index + 1)
#
#         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#             end_pointer = block_end if block_end is not None else content_len
#             futures = {executor.submit(process_row, (i, row)): i for i, row in
#                        dataset.loc[start_pointer:end_pointer].iterrows()}
#
#             results = {}
#             next_index_to_write = start_pointer
#
#             for future in concurrent.futures.as_completed(futures):
#                 i, input_text, input_result, target_text, target_result = future.result()
#                 results[i] = (input_text, input_result, target_text, target_result)
#
#                 while next_index_to_write in results:
#                     row_data = results.pop(next_index_to_write)
#                     writer.writerow([next_index_to_write] + list(row_data))
#                     utils.update_integer_in_file(start_pointer_file_path, next_index_to_write + 1)
#
#                     current_time = utils.get_current_time()
#                     speed = utils.get_speed(next_index_to_write - start_pointer, start_time, current_time)
#                     estimated_time = utils.get_estimated_time(content_len - start_pointer,
#                                                               i - start_pointer, start_time,
#                                                               current_time)
#
#                     mth.safe_print(
#                         f"Translated {next_index_to_write + 1} of {content_len}, Elapsed (Secs): {current_time - start_time}, Estimated (Hrs): {estimated_time}, Speed: {speed}")
#
#                     next_index_to_write += 1
#
#
# translate_dataset()

# Populate spreadsheet
# from openpyxl.utils import get_column_letter
# from openpyxl.styles import Alignment
# from openpyxl import Workbook
#
# wb = Workbook()
# ws = wb.active
#
# for col_num in range(1, 8):
#     col_letter = get_column_letter(col_num)
# ws.column_dimensions[col_letter].width = 30
#
# # Create data while iterating over the translated content
# data = [["inputs_en", "inputs_si", "inputs_nllb", "targets_en", "targets_si", "targets_nllb"]]
# for i in range(0, len(translated_content.translations), 2):
#     data.append([
#         content_list[i],
#         translated_content.translations[i].translated_text,
#         compare_content_list[i],
#         content_list[i + 1],
#         translated_content.translations[i + 1].translated_text,
#         compare_content_list[i + 1]
#     ])
#
# # Populate the worksheet with data
# for row in data:
#     ws.append(row)
#
# # Set the height of all rows to 100
# for row in ws.iter_rows():
#     if
# row[0].row == 1:
# continue
# for cell in row:
#     cell.alignment = Alignment(wrapText=True, vertical="top")
#     ws.row_dimensions[cell.row].height = 100
#
# # Save the workbook
# wb.save(config.SPREADSHEET_NAME)
# print(f"Excel file '{config.SPREADSHEET_NAME}' created and populated successfully.")
