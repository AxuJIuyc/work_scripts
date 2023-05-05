import pika
from datetime import datetime
from utils.time_utils import get_time
from utils.formatting import to_labelme
import json


class RabbitSender():
    def __init__(self):
        self.time_start = None
        self.ready = False
        self.time_old = 0
        self.time_now = 0
    
    # Wait 1 second before sending messages if target class is available
    def wait_start(self, available):
        if available:
            if not self.ready:
                if self.time_start is None:
                    self.time_start = datetime.now().second
                elif abs(datetime.now().second - self.time_start) > 3:
                    self.ready = True
                    print("--- - - - - - - - -  ---")
                    print("--- Target is fixed! ---")
                    print("--- - - - - - - - -  ---")

        else:
            self.time_start = None
            self.ready = False
            print("--- Wait RESTART ---")

    # Checking if the target is within the recognition boundaries
    def target_place(self, target_place, class_position):
        tp = target_place
        cp = class_position
        if (cp[0] >= tp[0] 
        and cp[1] >= tp[1] 
        and cp[2] <= tp[2] 
        and cp[3] <= tp[3]
        ):
            return True
        else:
            return False
    
    # 3 second pause between messages
    def pause(self):
        self.time_now = datetime.now().second
        if abs(self.time_now - self.time_old) < 3:
            return False
        else:
            self.time_old = self.time_now
            return True
    
    # Sending function
    # Now for each message a transmission channel is created and closed. 
    # Later, it will be possible to separate the operations of creating
    # a channel and passing messages to different functions.
    def send_rabbit(self,
                    text='void', 
                    queue='camera_data', 
                    ip='192.168.193.209', 
                    port=5672, 
                    host='test_host', 
                    user='test_user', 
                    passw='user'
        ):
        if self.ready:
            # Set rabbit parameters
            credentials = pika.PlainCredentials(user, passw)
            parameters = pika.ConnectionParameters(ip, port, host, credentials)
            connection = pika.BlockingConnection(parameters)
            
            # Connection to channel
            channel = connection.channel()
            channel.queue_declare(queue=queue)
            
            # Sending the message
            # text = f"<{str(datetime.now())}>" + text
            channel.basic_publish(exchange='', routing_key=queue, body=text)
            
            # Check if the message has been delivered 
            # if channel.basic_ack():
            #     print('Sending ok')
            # else:
            #     print('Sending not ok')

            connection.close()

    # Check intresting class in detects list
    def check_target(self, dets, target=0):
        clss = dets[:, 4]
        print('===>dets:', clss)
        if [target] in clss:
            print('Target IS HERE')
            return True
        else:
            return False

# Convert data to labelme format and build json obj
def json_builder(frame, det, classes):
    height, width = frame.shape[0], frame.shape[1]
    name = get_time(0)
    text = to_labelme(content=(frame, det),
                    classes=classes,
                    frame_size=(width, height),
                    image_path=name)
    json_obj = json.loads(text)
    return json_obj