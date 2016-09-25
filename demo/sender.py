from django.conf import settings
from demo.utils import log_to_terminal

import os
import pika
import sys
import json


def vqa_task(image_path, question, socketid):

    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=settings.PIKA_HOST))
    channel = connection.channel()

    channel.queue_declare(queue='vqa_task_queue', durable=True)
    message = {
        'image_path': image_path,
        'question': question,
        'socketid': socketid,
    }
    log_to_terminal(socketid, {"terminal": "Publishing job to VQA Queue"})
    channel.basic_publish(exchange='',
                      routing_key='vqa_task_queue',
                      body=json.dumps(message),
                      properties=pika.BasicProperties(
                         delivery_mode = 2, # make message persistent
                      ))

    print(" [x] Sent %r" % message)
    log_to_terminal(socketid, {"terminal": "Job published successfully"})
    connection.close()
