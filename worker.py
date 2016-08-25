from __future__ import absolute_import
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vqa.settings')

from django.conf import settings
from demo.utils import log_to_terminal

import demo.constants as constants
import PyTorch
import PyTorchHelpers
import pika
import time
import yaml
import json

# Loading the VQA Model forever
VQAModel = PyTorchHelpers.load_lua_class(constants.VQA_LUA_PATH, 'HieCoattModel')
VqaTorchModel = VQAModel(
    constants.VQA_CONFIG['vqa_model'],
    constants.VQA_CONFIG['cnn_proto'],
    constants.VQA_CONFIG['cnn_model'],
    constants.VQA_CONFIG['json_file'],
    constants.VQA_CONFIG['backend'],
    constants.VQA_CONFIG['gpuid'],
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=settings.PIKA_HOST))

channel = connection.channel()

channel.queue_declare(queue='vqa_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):

    print(" [x] Received %r" % body)
    body = yaml.safe_load(body) # using yaml instead of json.loads since that unicodes the string in value

    result = VqaTorchModel.predict(body['input_image'], body['input_question'])
    result['input_image'] = str(result['input_image']).replace(settings.BASE_DIR, '')

    log_to_terminal(body['socketid'], {"terminal": json.dumps(result)})
    log_to_terminal(body['socketid'], {"result": json.dumps(result)})
    log_to_terminal(body['socketid'], {"terminal": "Completed VQA task"})

    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume(callback,
                      queue='vqa_queue')

channel.start_consuming()
