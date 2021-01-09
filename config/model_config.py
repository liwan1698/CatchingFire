
ROOT_PATH = '/Users/wanli/create/project/work/CatchingFire/'
STOP_WORDS_PATH = ROOT_PATH+'data/stopwords.txt'


class FastTextConfig:
    ORIGINAL_DATA_PATH = ROOT_PATH+'data/data.txt'
    PREPROCESS_PATH = ROOT_PATH+'data/preprocess_data.txt'
    VOCAB_PATH = ROOT_PATH+'data/vocab.txt'
    LABEL_PATH = ROOT_PATH+'data/label.txt'
    MODEL_SAVE_PATH = ROOT_PATH+'save_model/'
    SEQ_LENGTH = 100
    NUM_CLASSES = 10
    BATCH_SIZE = 32
    KEEP_PROB = 0.5
    EPOCHS = 1
    VOCAB_SIZE = 5000
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 300
    LEARNING_RATE = 1e-5
    TRAIN_TEST_SPLIT_VALUE = 0.9
