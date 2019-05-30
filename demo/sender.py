from django.conf import settings
from demo.utils import log_to_terminal

import os
import pika
import sys
import json


def vqa_task(image_path, question, vqa_model, socketid):

    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=settings.PIKA_HOST))
    channel = connection.channel()
    if vqa_model == "HieCoAtt":
        queue = "vqa_demo_task_queue"
        channel.queue_declare(queue=queue, durable=True)
    if vqa_model == "pythia":
        queue = "vqa_demo_task_queue_with_pythia"
        channel.queue_declare(queue=queue, durable=True)
    message = {
        'image_path': image_path,
        'question': question,
        'socketid': socketid,
    }
    log_to_terminal(socketid, {"terminal": "Publishing job to VQA Queue"})
    channel.basic_publish(exchange='',
                      routing_key=queue,
                      body=json.dumps(message),
                      properties=pika.BasicProperties(
                         delivery_mode = 2, # make message persistent
                      ))

    print(" [x] Sent %r" % message)
    log_to_terminal(socketid, {"terminal": "Job published successfully"})
    connection.close()
