from __future__ import absolute_import
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vqa.settings')

import django
django.setup()

from django.conf import settings
from demo.utils import log_to_terminal
from demo.models import QuestionAnswer

import demo.constants as constants
import PyTorch
import PyTorchHelpers
import pika
import time
import yaml
import json
import traceback
import signal
import requests
import atexit

django.db.close_old_connections()

# Loading the VQA Model forever
VQAModel = PyTorchHelpers.load_lua_class(constants.VQA_LUA_PATH, 'VQADemoTorchModel')

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

channel.queue_declare(queue='vqa_demo_task_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')


def handle_exit():
    print "Process killed. Sending log to Slack....."
    slack_data = {'text': "VQA demo not working!"}
    webhook_url = constants.SLACK_WEBHOOK_URL
    response = requests.post(
        webhook_url, data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )

atexit.register(handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def callback(ch, method, properties, body):

    print(" [x] Received %r" % body)
    body = yaml.safe_load(body) # using yaml instead of json.loads since that unicodes the string in value

    try:
        result = VqaTorchModel.predict(body['image_path'], body['question'])
        top5_answer = result['top5_answer']
        top5_softmax = result['softmax_score']
        top5_list = []
        for i in xrange(5):
            temp = {}
            temp['answer'] = top5_answer[i+1]
            temp['confidence'] = top5_softmax[i+1]*100
            top5_list.append(temp)

        top5_list = sorted(top5_list, key=lambda k: k['confidence'], reverse=True)
        result['top5_list'] = top5_list
        result['top5_answer'] = None
        result['softmax_score'] = None
        log_to_terminal(body['socketid'], {"terminal": json.dumps(result)})
        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        log_to_terminal(body['socketid'], {"terminal": "Completed VQA task"})
        try:
            QuestionAnswer.objects.create(question=body['question'], image=body['image_path'].replace(settings.BASE_DIR, ""), top5_answer=result['top5_list'], socketid=body['socketid'])
        except:
            print str(traceback.print_exc())

        django.db.close_old_connections()

    except Exception as e:
        print str(e)

    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume(callback,
                      queue='vqa_demo_task_queue')

channel.start_consuming()
