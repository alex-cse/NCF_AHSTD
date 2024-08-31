import logging
import configparser

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config
config = load_config('config.ini')

num_epochs = config.getint('TrainingParameters','num_epochs')
lr = config.getfloat('TrainingParameters','lr')
batch_size = config.getint('TrainingParameters','batch_size')
embedding_dim = config.getint('ModelParameters','embedding_dim')
hidden_dim = config.getint('ModelParameters','hidden_dim')
task_name = config['Task']['data_name']

# Set up logging automatically when this module is imported
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a console handler and set level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)




# Create a file handler and set level to INFO
log_file_name = f'log_{task_name}_embed{embedding_dim}_epoch{num_epochs}_batch{batch_size}.log'
file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
