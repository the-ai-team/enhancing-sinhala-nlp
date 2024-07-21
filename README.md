This script loads a dataset and translates the content to the Sinhala language row by row. It is utilized to split
content into chunks or combine it into blobs and return the translated content as necessary.

# File Structure

- dataset
- outputs
- keys
    - service-account.json (GCP) # For the moment, we don't use Google Cloud
- translate_content.ipynb <- Run this to translate the content from dataset to outputs
