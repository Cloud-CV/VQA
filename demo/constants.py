from django.conf import settings
import os


COCO_IMAGES_PATH = os.path.join(settings.MEDIA_ROOT, 'val2014')
COCO_IMAGES_URL = os.path.join(settings.MEDIA_URL, 'val2014')

VQA_LUA_PATH = "vqa.lua"

VQA_CONFIG = {
    'vqa_model' : 'models/model_alternating_train_vgg.t7',
    'cnn_proto' : 'image_model/VGG_ILSVRC_19_layers_deploy.prototxt',
    'cnn_model' : 'image_model/VGG_ILSVRC_19_layers.caffemodel',
    'json_file' : 'data/vqa_data_prepro.json',
    'backend' : 'cudnn',
    'gpuid' : 1,
    'image_dir': os.path.join(settings.MEDIA_ROOT, 'demo')
}
