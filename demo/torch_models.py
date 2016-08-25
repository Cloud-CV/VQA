import demo.constants as constants

import PyTorch
import PyTorchHelpers

# Loading the VQA Model forever
VQAModel = PyTorchHelpers.load_lua_class(constants.VQA_LUA_PATH, 'VQATorchModel')
VqaTorchModel = VQAModel(
    constants.VQA_CONFIG['vqa_model'],
    constants.VQA_CONFIG['cnn_proto'],
    constants.VQA_CONFIG['cnn_model'],
    constants.VQA_CONFIG['json_file'],
    constants.VQA_CONFIG['backend'],
    constants.VQA_CONFIG['gpuid'],
)
