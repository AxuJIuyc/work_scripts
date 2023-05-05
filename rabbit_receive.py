#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
 
import pika
import time
from datetime import datetime
from convert_img import img_b64_to_arr
from convert_img import img_data_to_arr
import cv2
import json

def get_time(time_format):
    """
    Get time in predefined formats
        Parameters:
            time_format(int) - Time format number
        Returns:
            time(str) - Time formatted into string
    """
    if time_format == 0:
        f = '%Y-%m-%d.%H-%M-%S.%f'
    elif time_format == 1:
        f = '%Y-%m-%d %H:%M:%S'
    elif time_format == 2:
        f = '%Y-%m-%d %H:%M'
    elif time_format == 3:
        f = '%Y-%m-%d.%H-%M-%S.%f'
        return datetime.now().strftime(f)[:-3]
    else:
        f = "%m.%d:%H.%M"
    return datetime.now().strftime(f)

credentials = pika.PlainCredentials('test_user', 'user')
parameters = pika.ConnectionParameters('localhost', 5672, 'test_host', credentials)
connection = pika.BlockingConnection(parameters)
queue = 'camera_data'
channel = connection.channel()
channel.queue_declare(queue=queue)

def callback(ch, method, properties, body):
    time = get_time(3)
    # print(time, "Received %r" % body)
    print(time, "message received")
    with open(f'frutilad_data/{time}.json', 'w') as f:
        body = str(body)
        body = body.replace('\\\\', '\\')
        body = body.replace('b\"{', '{')
        body = body.replace('}\"', '}')
        body = body.replace('\'', '\"')
        body = body.replace('None', 'null')
        json_obj = json.loads(body)
        json.dump(json_obj, f)

channel.basic_consume(queue=queue, auto_ack=True, on_message_callback=callback)
print('start consuming')
channel.start_consuming()
print('after start consuming')
channel.stop_consuming()
print('To exit press CTRL+C')
connection.close()
