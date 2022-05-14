import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ = 75
EMBED_DIMS = 64
ENC_HEADS = DEC_HEADS = 2
NUM_ENCODER = NUM_DECODER = 6
BATCH_SIZE = 128
TRAIN_FILE = "/opt/ml/input/data/processed_data.csv"
TOTAL_EXE = 9455
TOTAL_CAT = 1538
TOTAL_TAG = 913
TOTAL_CHAP = 448
TOTAL_TEST = 199
model = "sakt"
n_epochs = 30