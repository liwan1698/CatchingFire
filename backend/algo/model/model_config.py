ROOT_PATH = '/Users/wanli/create/project/CatchingFire/'
CLASSIFY_DATA_PATH = ROOT_PATH+'backend/data/classify/'
MODEL_SAVE_PATH = ROOT_PATH+'backend/save_model/'

class FastTextConfig:
    STOP_WORDS_PATH = CLASSIFY_DATA_PATH+'stopwords.txt'
    ORIGINAL_DATA_PATH = CLASSIFY_DATA_PATH+'data.txt'
    PREPROCESS_PATH = CLASSIFY_DATA_PATH+'preprocess_data.txt'
    VOCAB_PATH = CLASSIFY_DATA_PATH+'vocab.txt'
    LABEL_PATH = CLASSIFY_DATA_PATH+'label.txt'
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
    MODEL_SAVE_PATH = MODEL_SAVE_PATH+'classify/'


class BertBilstmCrfConfig:
    BERT_MODEL_DIR = ROOT_PATH+'backend/save_model/chinese_L-12_H-768_A-12/'
    MSRA_DIR = ROOT_PATH+'backend/data/MSRA'
    VOCAB_PATH = ROOT_PATH+'backend/save_model/chinese_L-12_H-768_A-12/vocab.txt'


class BertRelationExtract:
    BERT_MODEL_DIR = ROOT_PATH+'backend/save_model/chinese_L-12_H-768_A-12/'
    CONFIG = BERT_MODEL_DIR+'bert_config.json'
    CHECK_POINT = BERT_MODEL_DIR+'bert_model.ckpt'
    VOCAB_PATH = BERT_MODEL_DIR+'vocab.txt'
