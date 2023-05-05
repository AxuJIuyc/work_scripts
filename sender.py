import pika
from convert_img import img_arr_to_b64
from convert_img import img_to_file


class RabbitSender():    
    counter = 0

    def __init__(self):
        pass
    def send_rabbit(self,
                    text='void', 
                    queue='camera_data', 
                    ip='192.168.193.209', 
                    port=5672, 
                    host='test_host', 
                    user='test_user', 
                    passw='user'
        ):
        """
        Sends the "text" of the message to the "queue" of messages to the specified "ip"
        address and "port" of the virtual "host" for the "user" with the "password".
        """
        credentials = pika.PlainCredentials(user, passw)
        parameters = pika.ConnectionParameters(ip, port, host, credentials)
        connection = pika.BlockingConnection(parameters)
        
        channel = connection.channel()
        # Turn on delivery confirmations
        channel.confirm_delivery()
        
        # Send a message
        channel.queue_declare(queue=queue) # Declare queue, create if needed.
        # text = img_arr_to_b64(text)
        # text = img_to_file(text)
        channel.basic_publish(exchange='first_test', routing_key='testing', body=text)
        # try:
        #     channel.basic_publish(exchange='first_test',
        #                         routing_key='testing',
        #                         body=text,
        #                         properties=pika.BasicProperties(content_type='text/plain',
        #                                                         delivery_mode=pika.DeliveryMode.Transient)
        #                         )
        #     print('Message publish was confirmed')
        # except pika.exceptions.UnroutableError:
        #     print('Message could not be confirmed')
        # Check if the message was delivered

        if channel.basic_ack():
            print("Message delivered")
        else:
            print("Message not delivered")

        connection.close()

    def notice(self, channel):
        """
        Verifying that messages have been delivered.

        :param pika.BlockingConnection.channel

        :return bool
        """
        notice = channel.confirm_delivery()
        if notice:
            return True
        else:
            return False
            
        

def main():
    sender = RabbitSender()
    sender.send_rabbit()

if __name__ == "__main__":
    main()