##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

##########################################################################################
# import

import logging
from utils.utils import create_logger

from TSPTester import TSPLIBTester as Tester


##########################################################################################
# parameters

logger_params = {
    'log_file': {
        'tag': 'test_n100_on_tsplib', # tag for log file name
        'filename': 'log.txt'
    }
}

model_params = {
    'start_node': 'pomo', # pomo rollout
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_n100',  # directory path of pre-trained model and log files saved.
        'epoch': 700,  # epoch version of pre-trained model to laod.
    },
    'augmentation_enable': True,
    'aug_factor': 8,
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(model_params=model_params,
                    tester_params=tester_params)

    tester.run()


def _set_debug_mode():
    global tester_params


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
