import shutil
import zipfile

from chilly.data import generate_dataset


RAW_DATA_FILE = "data/slack.zip"
RAW_DATA_DIR = "data/raw_data"
TRAIN_LEN = 100000
VAL_LEN = 1000


# Generate raw data
shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)
with zipfile.ZipFile(RAW_DATA_FILE, "r") as raw_data_file:
    raw_data_file.extractall(RAW_DATA_DIR)
print("Number of conversations: ", generate_dataset(RAW_DATA_DIR, TRAIN_LEN, VAL_LEN))
