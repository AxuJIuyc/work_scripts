from rabbit_test_send_1 import send_rabbit
from sender import RabbitSender


counters = {
    "hostname": 'self.hostname',
    "FPS": 'round(self.fps.value, 2)',
    "start_time": 'self.start_time',
    "current_time": 'get_time(3)',
    #"counted_this_iteration": counted_lists,
    "counters": {'car': 2, 'maps': 3}
}


# text = 'test_data/2.json'
# with open(text, 'r') as f:
#     # send_rabbit(str(counters))
#     send_rabbit(f)

sender = RabbitSender()
for i in range(20):
    print('print:', sender.fnum)
    sender.send_rabbit(f'text-{i}')