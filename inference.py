import os
import time
import cv2
import traceback
import numpy as np
from multiprocessing import Process, Manager, Value, Lock
from multiprocessing.managers import BaseManager
from threading import Thread
from copy import copy, deepcopy
from zipfile import ZipFile

from byte_tracker import BYTETracker, BTArgs

from settings import load_settings, save_corrected_settings
from constants import *
from utils import *  # Refer utils.__init__.py for exact imports
from c_api import libc

from telegram import Bot

# from test_send import send_rabbit
from utils.sender import RabbitSender, json_builder

ROOT = os.path.dirname(os.path.abspath(__file__))


def check_model(model):
    '''
    Check if loaded model is appliable
    Parameters:
        model(str) - path to model
    Returns:
        bool - True if model is valid else False
    '''
    res = Value('i', 1)

    def inner(mdl):
        context = libc.create_context("timvx".encode('utf-8'), 1)
        libc.init_tengine()
        libc.set_context_device(context, "TIMVX".encode('utf-8'), None, 0)
        libc.create_graph(context, "tengine".encode('utf-8'), mdl.encode('utf-8'))
        res.value = 0

    p = Process(target=inner, args=(model,))
    p.start()
    p.join()
    return not bool(res.value)


class Inference:
    def __init__(self):
        self.create_shared_buffers() # Creating arrays with shared buffers
        BaseManager.register('Message', Message) # Registering classes to be used
        BaseManager.register('VideoCapture', cv2.VideoCapture) # across processes
        self.base_manager = BaseManager()
        self.base_manager.start()

        self.manager = Manager()
        self.settings = self.manager.dict() # This dictionary contains user settings
        self.messages = self.manager.list() # List for messages 

        self.lock = Lock()  # Process Lock for predicatble memory managment

        #self.counted_this_iteration = self.manager.list()
        self.counters = self.manager.dict() # Dictionary contains classes counters numbers {class: number_of_counted_items}
        self.counters_sets = {}     # Dictionary contains sets of track IDs to count items
        self.counters_archived_values = {}  # Dictionary contains number of counted items which IDs was cleared from sets

        self.flags[MODEL_UPD] = 0
        self.cap = None
        self.fps_msg = self.base_manager.Message(text='', expire_after=-1)              # Defining messages to be used
        self.error_msg = self.base_manager.Message(text='Bad source!', expire_after=3)
        self.updating_model_msg = self.base_manager.Message(text='', expire_after=-1)

        self.update_settings()  # Loading settings from ./settings/settings.json
        self.flags[STOP] = 1
        self.messages.insert(0, self.fps_msg)       # Adding messages to list of messages to be drawn
        self.messages.insert(1, self.error_msg)
        self.messages.insert(2, self.updating_model_msg)

        # Preprocessor process reads cam and formatting image to appropriate format
        self.preprocesser = Process(target=self.read_cam, daemon=True)
        
        # FPS value
        self.fps = Value('d', 0)
        # Cleaner thread used to clean old data and submit new to variables
        self.cleaner = Thread(target=self.clean_and_update_data, daemon=True)
        # Loading model implies defining NUM_PROC number of graphs to be run inference on
        # Processes stored in self.inferencers[]
        self.load_model(os.path.join(ROOT, MODEL), change=False)
        

        # Debug informer used to print statuses and indicies 
        self.debug_informer = Process(target=self.print_debug_info, daemon=True)
        # Visualiser process draws gathered information on frame and puts in into shared self.to_show buffer 
        self.visualiser = Process(target=self.visualise, daemon=True)
        # Telegram notificator helps monitoring device state by sending image and counters every 10 minutes
        self.notificator = Process(target=self.notificate, daemon=True)
        
        byte_tracker_args = BTArgs()    # Byte Tracker arguments
        self.byte_tracker = BYTETracker(byte_tracker_args, frame_rate=20) # Byte tracker to perform tracking
        # Start time date
        self.start_time = get_time(3)
        self.hostname = os.uname()[1]

        # Postman's announcement
        self.sender = RabbitSender()

    def create_shared_buffers(self):
        # Creating shared buffers
        self.buffers = {
            "preprocessed_frames": create_shared_memory(size=BUF_SZ * 3 * 352 * 352, name="preprocessed_frames"),
            "raw_frames": create_shared_memory(size=BUF_SZ * 3 * 480 * 640, name="raw_frames"),
            "detections": create_shared_memory(size=BUF_SZ * 4 * NUM_DETS * 6, name="detections"),
            "tracks": create_shared_memory(size=BUF_SZ * 4 * NUM_DETS * 6, name="tracks"),
            "status": create_shared_memory(size=BUF_SZ, name="status"),

            "to_show": create_shared_memory(size=480 * 640 * 3, name="to_show"),
            "last_frame": create_shared_memory(size=480 * 640 * 3, name="last_frame"),
            "last_data": create_shared_memory(size=2 * 4 * NUM_DETS * 6, name="last_data"),

            "indicies": create_shared_memory(size=8 * 4 + 8 * NUM_PROC, name="indicies"),

            "flags": create_shared_memory(size=3, name="flags")
        }
        # Preprocessed frames array contains frames to be sent to inference
        self.preprocessed_frames = np.ndarray([BUF_SZ, 3, 352, 352], dtype=np.uint8,
                                    buffer=self.buffers["preprocessed_frames"].buf)
        # Raw frames array contains frames from camera without preprocessing
        self.raw_frames = np.ndarray([BUF_SZ, 480, 640, 3], dtype=np.uint8,
                                    buffer=self.buffers["raw_frames"].buf)
        # Detections array contains inference data
        self.detections = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32,
                                    buffer=self.buffers["detections"].buf)
        # Tracks array contains tracking data
        self.tracks = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32,
                                    buffer=self.buffers["tracks"].buf)
        # Status array contains image status
        # 1 - Preprocessed
        # 2 - Being inferenced
        # 3 - Inferenced
        # 4 - Being postprocessed
        # >=5 - All done, may be rewritten
        self.status = np.ndarray([BUF_SZ], dtype=np.uint8,
                                    buffer=self.buffers["status"].buf)
        self.status[:] = np.full([BUF_SZ], 6, dtype=np.uint8)
        # To show array contains image to be shown in player or to be sent via bot
        self.to_show = np.ndarray([480, 640, 3], dtype=np.uint8,
                                    buffer=self.buffers["to_show"].buf)
        # Last frame array contains last postprocessed frame. Needed for Visualiser
        self.last_frame = np.ndarray([480, 640, 3], dtype=np.uint8,
                                    buffer=self.buffers["last_frame"].buf)
        # Last data array contains both last image detections and tracking data
        self.last_data = np.ndarray([2, NUM_DETS, 6], dtype=np.float32,
                                    buffer=self.buffers["last_data"].buf)
        # Indicies array contains indicies used by processes
        # 0 - Read index
        # 1 - Postprocess index
        # 2 - Last postprocessed index
        # 3 - Requested by request_inference image index
        # >=4 - Inference indicies 
        self.indicies = np.ndarray([4 + NUM_PROC], dtype=np.int64,
                                    buffer=self.buffers["indicies"].buf)
        self.indicies[:] = np.zeros([4 + NUM_PROC], dtype=np.int64)
        # 0 - Stop flag
        # 1 - Inference requested flag
        # 2 - Model being updated flag
        self.flags = np.ndarray([3], dtype=np.uint8,
                                    buffer=self.buffers["flags"].buf)
        self.flags[:] = np.asarray([1, 0, 0], dtype=np.uint8)

    def update_settings(self, settings_path=None):
        """
        Update settings
            Parameters:
                settings_path - path to settings.json file
        """
        while self.flags[MODEL_UPD] == 1: pass # Don't update settings while updating model
        settings_path = settings_path or os.path.join(ROOT, "settings/settings.json") # Set default path if not specified
        sets = load_settings(settings=settings_path) # Loading settings from file
        for st in sets:
            self.settings[st] = sets[st]
        self.flags[STOP] = 1    # Stop algorithm
        time.sleep(0.1)
        self.error_msg.set_color(colors[self.settings["colors"]["error"]])  # Updating messages color
        self.fps_msg.set_color(colors[self.settings["colors"]["info"]])
        self.fps_msg.set_display(self.settings["inference"]["print_fps"])
        self.settings["camera"] = self.adjust_cam(self.settings["camera"])  # Updating camera settings
        save_corrected_settings(self.settings)  # Saving corrected settings after applying
        self.flags[STOP] = 0    # Unstop algorithm

    def adjust_cam(self, camera_settings):
        """
        Set new camera settings and return data from camera
            Parameters:
                camera_settings(dict) - Dictionary containing camera settings
            Returns:
                camera_settings(dict) - Corrected settings retrieved from camera
        """
        if self.cap is None:
            self.cap = self.base_manager.VideoCapture(camera_settings["source"])
        else:
            self.cap.open(camera_settings["source"])
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        # Setting parameters to VideoCapture object
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_settings["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_settings["height"])
        # Getting actual parameters from cam
        camera_settings["width"] = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        camera_settings["height"] = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.cap.set(cv2.CAP_PROP_FPS, camera_settings["fps"])
        self.cap.set(cv2.CAP_PROP_SATURATION, camera_settings["saturation"])
        self.cap.set(cv2.CAP_PROP_HUE, camera_settings["hue"])
        self.cap.set(cv2.CAP_PROP_CONTRAST, camera_settings["contrast"])
        self.cap.set(cv2.CAP_PROP_GAMMA, camera_settings["gamma"])
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, camera_settings["brightness"])
        camera_settings["fps"] = int(self.cap.get(cv2.CAP_PROP_FPS))
        camera_settings["saturation"] = int(self.cap.get(cv2.CAP_PROP_SATURATION))
        camera_settings["contrast"] = int(self.cap.get(cv2.CAP_PROP_CONTRAST))
        camera_settings["hue"] = int(self.cap.get(cv2.CAP_PROP_HUE))
        camera_settings["gamma"] = int(self.cap.get(cv2.CAP_PROP_GAMMA))
        camera_settings["brightness"] = int(self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
        return camera_settings

    def load_model(self, model, change=True):
        """
        Check model, create inferencers processes and load model to them
            Parameters:
                model(str)/model(bytes) - Path to model file or model in bytes format
                change(bool) - Defines if model is loaded first time or being changed
        """
        if not self.flags[MODEL_UPD]: # Check if model is loading at the time
            self.flags[MODEL_UPD] = 1   # Set MODEL_UPD flag to 1 (being updated)
            self.updating_model_msg.set_text('Loading model...')    # Changing updating_model_msg text, color
            self.updating_model_msg.set_color(colors[self.settings['colors']['info']])
            self.updating_model_msg.renew_exp(-1) # Setting to infinite display
            self.updating_model_msg.set_display(True)
            print('Loading model...')
            
            nm_path = os.path.join(ROOT, 'new_model.tmfile') # New model file path
            if not isinstance(model, str): # If model is not path, but bytes
                with open(nm_path, 'wb') as f:  
                    f.write(model)  # Write to file
                model_path = nm_path
            else:
                model_path = model
            
            if not check_model(model_path): # Check model
                # If model is bad, notify and end updating process
                print('Bad Model')
                self.updating_model_msg.set_text('Bad model.')
                self.updating_model_msg.set_color(color=colors[self.settings['colors']['error']])
                self.updating_model_msg.renew_exp(5)
                return
            # If model is valid 
            dm_path = os.path.join(ROOT, 'model.tmfile') # Default model path
            if model_path != dm_path: # If model is not default
                with open(dm_path, 'wb') as f:
                    f.write(model) # Write as new default model
            self.flags[STOP] = NUM_PROC # Setting stop flag equals to inferencers process number
            time.sleep(1)
            if change: # If model is already running
                self.postprocesser.kill()
                for i in self.inferencers:
                    i.kill()    # Kill old processes
            
            byte_tracker_args = BTArgs()    # Byte Tracker arguments
            self.byte_tracker = BYTETracker(byte_tracker_args, frame_rate=30) # Byte tracker to perform tracking
            # Postprocesser process does tracking and counting based on inference information
            time.sleep(1)
            # Initialize new processes
            self.postprocesser = Process(target=self.postprocess, daemon=True)
            self.inferencers = [Process(target=self.run_inference, args=(i, model_path)) for i in range(NUM_PROC)]
            for inf in self.inferencers:
                inf.start() # And start them
            self.postprocesser.start()
            self.flags[MODEL_UPD] = 2 # Set flag to updated (needed by cleaner thread)

    def read_cam(self):
        """
        Preprocesser process main thread. Reads camera and preprocessing frames for inference
        """
        blank_frame = np.zeros([480, 640, 3], dtype=np.uint8) # Blank frame for bad source cases
        while True:
            try:
                if not self.flags[STOP]: # If algorithm is running
                    idx = self.indicies[READ] % BUF_SZ # Get corresponding index
                    while self.status[(self.indicies[READ]) % BUF_SZ] < 5: # Wait to be processed
                        pass
                    ret, frame = self.cap.read() # Read camera
                    if frame is None: # If cap returned None
                        frame = blank_frame # Set frame to blank
                        if self.error_msg not in self.messages: # Set error_msg to be displayed
                            self.error_msg.renew_exp()
                            self.error_msg.set_display(True)

                    if frame.shape[0] != 480: # Resize input frame if not in desired format (DOESN'T WORK NOW, TODO)
                        frame = cv2.resize(frame, [640, 480], interpolation=cv2.INTER_LINEAR)
                    with self.lock: # Locking memory
                        # Writing data to raw_frames, preprocessed_frames cells, emptying detections cell and setting status to 1 (Preprocessed)
                        self.raw_frames[idx][:], self.detections[idx][:],                    self.preprocessed_frames[idx][:],       self.status[idx] = \
                        frame,                    np.zeros([NUM_DETS, 6], dtype=np.float32), letterbox(frame, new_shape=(352, 352)), 1
                    self.indicies[READ] += 1 # Setting index to next one
            except:
                print(traceback.format_exc())


    def run_inference(self, proc, model):
        """
        Inferencer process main thread
            Parameters:
                proc(int) - Inferencer process id
                model(str) - Path to model file
        """
        # Initializing graphs and nodes
        self.img_sz = 352
        context = libc.create_context("timvx".encode('utf-8'), 1)
        libc.init_tengine()
        libc.set_context_device(context, "TIMVX".encode('utf-8'), None, 0)
        model = model.encode('utf-8')
        self.graph = libc.create_graph(context, "tengine".encode('utf-8'), model)
        libc.set_graph(352, self.img_sz, self.graph)
        self.input_tensor = libc.get_graph_input_tensor(self.graph, 0, 0)
        self.output_node_num = libc.get_graph_output_node_number(self.graph)
        
        self.dets = np.zeros([NUM_DETS, 6], dtype=np.float32)
        self.classes = libc.get_classes(self.graph)
        if not self.indicies[INF + proc]: 
            self.indicies[INF + proc] = proc
        print("Initialised", proc)
        self.flags[STOP] -= 1 # Decreasing stop flag for each process
        if not self.flags[STOP]: # If all processes are loaded
            # Notify user
            self.updating_model_msg.set_text('Model loaded successfully!')
            self.updating_model_msg.set_color(colors[self.settings['colors']['info']])
            self.updating_model_msg.renew_exp(5)
            print('Model loaded')
        while True:
            if not self.flags[STOP]: # If algorithm is running
                self.do_inference(proc)

    def do_inference(self, proc):            
        idx = self.indicies[INF + proc] % BUF_SZ # Get corresponding index
        while self.status[idx] != 1: # Waiting for image to be preprocessed
            pass
        self.status[idx] = 2  # Setting status to being inference
        frame = self.preprocessed_frames[idx] # Getting preprocessed frame
        # Run inference
        libc.set_image_wrapper(frame.ctypes.data, 352, 352, self.input_tensor, self.img_sz, self.img_sz)
        libc.run_graph(self.graph, 1)
        libc.postprocess_wrapper(480, 640, self.dets.ctypes.data,
                                            self.graph, self.output_node_num, 352, self.img_sz, self.classes, NUM_DETS,
                                            self.settings["inference"]["nms"], self.settings["inference"]["threshold"])
        self.detections[idx][:] = self.dets[:] # Set inference data to corresponding cell
        self.status[idx] = 3  # Set status to inferenced
        self.indicies[INF + proc] += NUM_PROC # Setting index to next one for that process

    def start(self):
        """
        Start all processes
        """
        self.preprocesser.start()
        #self.debug_informer.start()
        self.visualiser.start()
        self.notificator.start()

    def postprocess(self):
        """
        Postprocesser process main thread
        """
        self.cleaner.start() # Starting cleaner in current process needed to make local variables accessible
        while True:
            if not self.flags[STOP]: # If algorithm is running
                self.do_postprocess()
    #@timeit
    def do_postprocess(self):
        idx = self.indicies[POSTPROCESS] % BUF_SZ # Get corresponding index
        while self.status[idx] != 3:    # Wait for status to become inferenced
            pass
        
        frame, dets = copy(self.raw_frames[idx]), copy(self.detections[idx]) # Copying data needed for tracking
        
        filtered_dets = self.filter_dets(dets) # Filter detections
        
        if filtered_dets is not None: # If there are detections after filtering
            output = self.do_tracking(frame.shape, filtered_dets) # Do tracking
        else:
            output = None
        
        self.tracks[idx] = np.zeros([NUM_DETS, 6], dtype=np.float32) # Empty tracking info
        if output is not None: # If there is output from tracking
            self.tracks[idx][:output.shape[0], :output.shape[1]] = output # Set tracking data to array
            if self.settings["counting"]["do_counting"]:
                this_frame_tracks = self.do_counting(output)    # Count items based on tracking output
                #if len(this_frame_tracks) > 0:
                #    self.counted_this_iteration.append([self.indicies[POSTPROCESS]] + this_frame_tracks)
        with self.lock: # Locking memory 
            # Set new data to arrays
            self.last_frame[:], self.last_data[0],  self.last_data[1],       self.indicies[LAST] = \
            frame[:],           dets[:],            copy(self.tracks[idx]), self.indicies[POSTPROCESS]
            # Set status to all done or to be postprocessed if inference is requested
            self.status[idx] = 5 - self.flags[INF_REQ]
        # Setting index to next item
        self.indicies[POSTPROCESS] += 1

    def filter_dets(self, dets):
        """
        Filter zero scored detections and that are not in to_display list
            Parameters:
                dets(numpy.ndarray) - Detections to be filtered
            Returns:
                dets(numpy.ndarray) - Filtered detections
        """
        dets = dets[np.where(dets[..., 5] > 0)] # Filter detections with 0 score
        # Filter dets by to_display list
        dets = dets[np.where(np.isin(dets[..., 4], self.settings["classes"]["to_display"]))]
        if len(dets) == 0: # If no detections left
            return None # Return None
        # dets[:, 2:4] += dets[:, 0:2] # Convert xywh to x1y1x2y2
        return dets

    def do_tracking(self, frame_shape, dets):
        """
        Do tracking
            Parameters:
                frame(numpy.ndarray) - Current frame shape
                dets(numpy.ndarray) - Detections to be tracked
            Returns:
                output(numpy.ndarray) - Tracking output
        """
        if self.settings["tracking"]["tracker"] == 'byte': # If chosed tracker is ByteTracker (TODO ADD ANOTHER TRACKER(S))
            # Update tracker object and get output
            output = self.byte_tracker.update(dets, [frame_shape[0], frame_shape[1]], [frame_shape[0], frame_shape[1]])
            # Reformat from list of objects to list of [x1, y1, x2, y2, track_id, class_id]
            output = [np.append(out.tlbr, [out.track_id, out.sclass]) for out in output]
            if len(output):
                return np.asarray(output) # Convert list to numpy.ndarray
            else:
                return None
        else:
            return None

    def do_counting(self, output):
        """
        Count items
            Parameters:
                output(numpy.ndarray) - Tracking data
            Returns:
                counted_this_frame(int) - Number of items counted on this frame
        """
        # Retrieve boundaries from settings
        left_bound = self.settings["counting"]["left"]
        right_bound = self.settings["counting"]["right"]
        upper_bound = self.settings["counting"]["up"]
        bottom_bound = self.settings["counting"]["down"]
        # Retrieve classes list
        clss_list = self.settings["classes"]["list"]
        # Convert x1y1 to xcyc
        output[:, 0:2] = (output[:, 0:2] + output[:, 2:4]) / 2
        # output = output[
        #     np.where(right_bound  > output[..., 0] > left_bound \
        #         and  bottom_bound > output[..., 1] > upper_bound)]
        for out in output:
            if not (right_bound  > out[0] > left_bound and \
                    bottom_bound > out[1] > upper_bound):
                continue
            cls_ = clss_list[int(out[5])] # Get class name
            try:
                self.counters_sets[cls_].add(out[4]) # Add track id to set of counted ids set
            except KeyError: # If there's no set of ids for that class
                self.counters_sets[cls_] = set() # Make one
                self.counters_archived_values[cls_] = 1  # And count that item'''

    def visualise(self):
        """
        Visualisator process main thread
        """
        self.start_time = get_time(3) # Set start time to current
        last_drawn = 0 # Last drawn image id
        while True:
            if not self.flags[STOP]: # If algorithm is running
                # Get data to be shown
                frame, dets, tracks, last_idx = \
                    copy(self.last_frame), *(copy(self.last_data)), copy(self.indicies[LAST])
                # If this frame was not already processed
                if last_drawn != last_idx:
                    try:
                        # Draw all data on frame
                        to_show = self.draw_info_on_frame(frame, dets, tracks)
                        with self.lock: # With locking memory
                            self.to_show[:] = copy(to_show[:]) # Set new to_show frame
                    except:
                        print(traceback.format_exc())
                    last_drawn = last_idx # Update last drawn id

    def draw_info_on_frame(self, frame, dets, tracks):
        """
        Draws inference and tracking data on frame
            Parameters:
                frame(numpy.ndarray) - Frame to be drawn on
                dets(numpy.ndarray) - Detections to be drawn
                tracks(numpy.ndarray) - Tracking data to be drawn
            Returns:
                new_frame(numpy.ndarray) - Frame with drawn info
        """
        stock_dets = dets
        dets = self.filter_dets(dets) # Filter detections
        tracks = tracks[np.where(tracks[..., 4] > 0)] # Filter zero tracks
        new_frame = frame
        
        # If there are detections
        if dets is not None: 
            # If draw_detections is set to True
            if self.settings["inference"]["draw_detections"]:
                # Draw detections
                new_frame = self.draw_detections(dets, new_frame)
        # If there are tracks
        if len(tracks):
            # If draw_tracking is set to True
            if self.settings["inference"]["draw_tracking"]:
                # Draw tracking
                new_frame = self.draw_tracks(tracks, new_frame)
        # Update fps message with current fps
        self.update_fps_message()
        
        # Draw messages and return frame
        drawed_frame = self.draw_messages_on_img(new_frame)

        # Сheck target class  in frame and send message
        target_place = (50, 50, 590, 430)
        if dets is not None:
            target_det = list(map(int, dets[0, :4]))
        
            if (self.sender.check_target(dets, target=1) 
            and self.sender.target_place(target_place, target_det)
            ):
                self.sender.wait_start(True)
                if self.sender.pause():
                    message = json_builder(drawed_frame, 
                                        dets, 
                                        self.settings["classes"]["list"])
                    print("====> json sent")
                    self.sender.send_rabbit(str(message))
            else:
                self.sender.wait_start(False)
        cv2.rectangle(drawed_frame, target_place[:2], target_place[2:], [0, 0, 200], 3, 1)

        return drawed_frame


    
    def update_fps_message(self):
        """
        Update fps message text
        """
        if self.settings["inference"]["print_fps"]:
            self.fps_msg.set_text(f'{self.fps.value:.2f} FPS')
        else:
            self.fps_msg.set_display(False)

    def draw_detections(self, dets, img):
        """
        Draw detections on image
            Parameters:
                dets(numpy.ndarray) - Detections to be drawn
                img(numpy.ndarray) - Image to be drawn on
            Returns:
                img(numpy.ndarray) - Image with drawn detections
        """
        clr = colors[self.settings["colors"]["bbox"]]
        scl = self.settings["scale"]["id_scale"]
        space_scl = int(scl * default_bias)
        for det in dets:
            org = det[:4].astype(int)
            img = draw_box_on_img(img, clr, org)
            try:
                name = self.settings["classes"]["list"][int(det[4])]
            except:
                name = int(det[4])
            text = f'C: {name}\nS: {det[5]:.2f}'
            img = draw_text_on_img(img, text, clr, scl, org=(org[0] + 3, org[1]), space_scl=space_scl)
        return img

    def draw_tracks(self, tracks, img):
        """
        Draw tracks on image
            Parameters:
                tracks(numpy.ndarray) - Tracking data to be drawn
                img(numpy.ndarray) - Image to be drawn on
            Returns:
                img(numpy.ndarray) - Image with drawn tracks
        """
        clr = colors[self.settings["colors"]["tbox"]]
        scl = self.settings["scale"]["id_scale"]
        space_scl = int(scl * default_bias)
        for track in tracks:
            org = track[:4].astype(int)
            img = draw_box_on_img(img, clr, org)
            text = f'ID: {track[4]:.0f}'
            img = draw_text_on_img(img, text, clr, scl, org=(org[0] + 3, org[3] - space_scl), space_scl=space_scl)
        return img

    def draw_messages_on_img(self, img):
        """
        Draw messages on image
            Parameters:
                img(numpy.ndarray) - Image to be drawn on
            Returns:
                img(numpy.ndarray) - Image with drawn messages
        """
        accum_lines = 0 # Variable to store how many lines drawn
        for message in self.messages:
            if message.do_display(): # If message need to be displayed
                # Draw message on image
                org = np.asarray([10, accum_lines * default_bias * self.settings['scale']["text_scale"]]).astype(int)
                img = draw_text_on_img(img, message.get_text(), message.get_color(),
                                       self.settings['scale']["text_scale"], org=org)
                accum_lines += message.get_lines_count()
        return img

    def clean_and_update_data(self):
        """
        Clean, update data and calculate fps
        """
        last_pp = 0
        fps_calc_time = time.time()
        flag = True
        while True:
            start_time = time.time()
            # Calculate fps value
            self.fps.value = (self.indicies[POSTPROCESS] - last_pp) / (time.time() - fps_calc_time)
            # Remember last index fps was calculated
            last_pp = self.indicies[POSTPROCESS]
            # And time
            fps_calc_time = time.time()
            
            # Set expired messages to not display
            for msg in self.messages:
                if msg.do_display() and msg.get_exp() < time.time():
                    msg.set_display(False)
            
            # If model had been updated
            if self.flags[MODEL_UPD] == 2:
                # Reset counters
                self.counters_sets = {}
                self.counters_archived_values = {}
                self.counters.clear()
                self.flags[MODEL_UPD] = 0

            # For each counted class
            with self.lock:
                counter_keys = self.counters_sets.keys()
                #self.counted_this_iteration[:] = [i for i in self.counted_this_iteration if (self.indicies[POSTPROCESS] - i[0] < 60)]
            for cls_ in counter_keys:
                # If there are more than 30 track ids written
                if len(self.counters_sets[cls_]) > 30:
                    # Clear old track ids
                    arr = np.asarray(list(self.counters_sets[cls_]))
                    arr_mean = np.mean(arr)
                    len1 = len(arr)
                    arr = arr[np.where(arr > arr_mean)]
                    len2 = len(arr)
                    self.counters_sets[cls_] = set(arr)
                    # Add number of deleted ids to self.counters_archived_values
                    self.counters_archived_values[cls_] += len1 - len2
                # Update counters value
                self.counters[cls_] = len(self.counters_sets[cls_]) + self.counters_archived_values[cls_]
            while time.time() - start_time < 1: pass # Do this every second

    def print_debug_info(self):
        """
        Debug informer process main thread.
        Prints statuses and indicies
        """
        while True:
            string = ''
            for i, stat in enumerate(self.status):
                if i in self.indicies[INF:] % BUF_SZ:
                    st = f'{FNT.BLUE}{FNT.BOLD}{stat}{FNT.ENDC} '
                elif i == self.indicies[READ] % BUF_SZ:
                    st = f'{FNT.RED}{FNT.BOLD}{stat}{FNT.ENDC} '
                elif i == self.indicies[POSTPROCESS] % BUF_SZ:
                    st = f'{FNT.YELL}{FNT.BOLD}{stat}{FNT.ENDC} '
                else:
                    st = f'{stat} '
                string += st

            print(
                f'\r{string} {self.indicies[READ]} {self.indicies[INF:]} {self.indicies[POSTPROCESS]}'
                f' {self.indicies[LAST]} {self.indicies[REQ]} {self.flags}',
                end='')
            time.sleep(0.001)

    async def request_inference(self):
        """
        Request inference data
        """
        # If inference isn't requested already and model is not updating
        if not self.flags[INF_REQ] and self.flags[MODEL_UPD] != 1: 
            self.flags[INF_REQ] = 1 # Set inference required flag
            # Create array for frames to store
            requested_frames = np.ndarray([self.settings["inference"]["mass_inf_number"], 480, 640, 3], dtype=np.uint8)
            # Create list for detections to store
            requested_dets = [None for _ in range(self.settings["inference"]["mass_inf_number"])]
            # Set requested inference index equals to current postprocessing index
            self.indicies[REQ] = self.indicies[POSTPROCESS]
            # Set current lookup index equals to requested index
            curr_idx = self.indicies[REQ]
            # Initialize index data to be set to
            set_ind = 0
            # Set all "postprocessed" statuses to "all done"
            # Needed so the algorithm won't freeze
            self.status[np.where(self.status == 4)[0]] = 5
            # While not enough data gathered
            while set_ind < self.settings["inference"]["mass_inf_number"]:
                # Wait for data to be postprocessed
                while self.status[self.indicies[REQ] % BUF_SZ] != 4:
                    pass
                # If current frame is requested
                if curr_idx == self.indicies[REQ]:
                    # Copy data to local variables
                    requested_frames[set_ind], requested_dets[set_ind] = copy(
                        self.raw_frames[self.indicies[REQ] % BUF_SZ]), self.filter_dets(
                        copy(self.detections[self.indicies[REQ] % BUF_SZ]))
                    # Add 1 to setter index
                    set_ind += 1
                    # Add mass_inf_step to requested index
                    self.indicies[REQ] += self.settings["inference"]["mass_inf_step"]
                # Set frame status to "all done"
                self.status[curr_idx % BUF_SZ] = 5
                # Go to next image
                curr_idx += 1
            

            # Convert data to labelme format and zip it into archive
            height, width = requested_frames[0].shape[0], requested_frames[0].shape[1]
            zippath = "inference.zip"
            with ZipFile(zippath, 'w') as zip_file:
                for i in range(self.settings["inference"]["mass_inf_number"]):
                    name = get_time(0)
                    zip_file.writestr(f'{name}.png', img_to_file(requested_frames[i]))
                    zip_file.writestr(f'{name}.json', to_labelme(content=(requested_frames[i], requested_dets[i]),
                                                                 classes=self.settings["classes"]["list"],
                                                                 frame_size=(width, height), image_path=name))
            # Set all "postprocessed" statuses to "all done"
            # Needed so the algorithm won't freeze
            self.status[np.where(self.status == 4)[0]] = 5
            # Reset flag
            self.flags[INF_REQ] = 0
            return zippath

    def notificate(self):
        """
        Notificator process main thread
        """
        start_time = time.time()
        while True:
            while time.time() - start_time < 600: pass
            start_time = time.time()
            try:
                bot = Bot(token=self.settings["notifications"]["bot_token"])
            except Exception as e:
                print("Can't start bot:")
                print(e)
                continue
            
            counters = {
                "hostname": self.hostname,
                "start_time": self.start_time,
                "current_time": get_time(3)
            }
            for cls_ in self.counters.keys():
                counters[cls_] = self.counters[cls_]

            caption = dict_to_json(counters)
            with self.lock:
                img = copy(self.to_show[:])
            img_path = os.path.join(ROOT, 'img.png')
            cv2.imwrite(img_path, img)
            time.sleep(1)
            with open(img_path, 'rb') as f:
                photo = f.read()
            sent = False
            retries_left = 5
            while not sent and retries_left:
                try:
                    bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption)
                    sent = True
                except:
                    retries_left -= 1
                    time.sleep(1)
                    print(traceback.format_exc())

    async def get_frame(self):
        """
        Return current frame with all drawn info
            Returns:
                new_frame(numpy.ndarray) - Frame with drawn info
        """
        return self.to_show[:]

    async def get_counters(self):
        """
        Get counters dictionary with start time and current time
            Returns:
                counters(dict) - Dictionary contains counters for classes
        """
        
        #counted_lists = [[str(i[0]), [str(j) for j in i[1]]] for i in self.counted_this_iteration]
        counters = {
            "hostname": self.hostname,
            "FPS": round(self.fps.value, 2),
            "start_time": self.start_time,
            "current_time": get_time(3),
            #"counted_this_iteration": counted_lists,
            "counters": {}
        }
        for cls_ in self.counters.keys():
            counters["counters"][cls_] = self.counters[cls_]


        return counters


if __name__ == "__main__":
    khadas = Inference()
    khadas.start()
    while True:
        pass
