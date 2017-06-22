from queue import Queue
import datetime
import threading
from collections import deque
import werkzeug.serving
import os.path
import json
import uuid
import io
import time
import shlex
import subprocess
import platform
import robosims.config
import atexit
import signal
import os
import logging
from flask import Flask, request, make_response, render_template, send_file, abort, Response
from robosims.actions import ActionBuilder

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

if platform.system() == 'Linux':
    import xcffib
    import xcffib.xproto

from PIL import Image
import numpy as np

try:
    import mss.linux
except:
    pass

def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError as e:
        return False
    return True

def format_sse(**params):
    lines = ["%s: %s" % (k, v) for k, v in params.items()]
    return "\n".join(lines) + "\n\n"

class Event(object):

    def __init__(self, metadata, frame_id, frame, frame_depth = None, frame_flow = None):
        self.metadata = metadata
        self.frame_id = frame_id
        self.frame = frame
        self.frame_depth = frame_depth
        self.frame_flow = frame_flow


class Controller(object):

    def __init__(self, config_file):
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.server = None
        self.config_file = config_file
        self.pid = -1

    def reset(self):
        if self.server.state == 'waiting_for_request':
            self.request_queue.get()
        if not self.request_queue.empty():
            raise ValueError('request queue not empty after get')

        return self.step(ActionBuilder.reset())

    def step(self, action):
        self.response_queue.put(action)
        event = self.request_queue.get()
        if not self.request_queue.empty():
            raise ValueError('request queue not empty after get')

        return event

    def unity_command(self, config):

        # XXX should make the path absolute, at least resolve from the current location
        target_arch = platform.system()
        if target_arch == 'Darwin':
            executable_path = config['ROBOSIMS_ENV_BUILD_DARWIN']
            command = shlex.split("open -n %s" % executable_path)
        elif target_arch == 'Linux':
            executable_path = config['ROBOSIMS_ENV_BUILD_LINUX']
            # we can lose the executable permission when unzipping a build
            os.chmod(executable_path, 0o755)
            command = shlex.split("./%s" % executable_path)
        else:
            raise Exception('unable to handle target arch %s' % target_arch)

        return command

    def setup_window_children(self, display):
        conn = xcffib.connect(display)
        setup = conn.get_setup()
        display_info = display.split('.')
        if len(display_info) > 1:
            screen_index = int(display_info[1])
        else:
            screen_index = 0

        screen = setup.roots[screen_index]
        xproto = xcffib.xproto.xprotoExtension(conn)

        def window_children():
            d = {}
            for x in xproto.QueryTree(screen.root).reply().children:
                d[x - (x % 16)] = x
                # p = xproto.ListProperties(x).reply()
                # print(p.__dict__)
                # print(p.atoms)

            return d

        return window_children

    def find_xwindow_id(self, pid, wc, pre_children):

        window_id = None
        for i in range(100):
            time.sleep(0.2)
            if not process_alive(pid):
                raise Exception("process died %s " % pid)

            post_children = wc()
            # print("len {} wc {}".format(len(post_children), post_children))

            diff = len(post_children) - len(pre_children)
            if diff < 0 or diff > 1:
                raise Exception("window count went down or increased by greater than 1 %s" % diff)
            elif diff == 1:
                output = (set(post_children.keys()) - set(pre_children.keys()))
                (window_id,) = output
                break

        if not window_id:        
            raise Exception("couldn't find window id")

        return window_id

    def _start_thread(self, env, config, port=0, start_unity=True):
        # get environment variables
        # load default configs or read from yaml file


        if not start_unity:
            self.server.client_token = None

        host, port = self.server.wsgi_server.socket.getsockname()
        env['ROBOSIMS_PORT'] = str(port)
        env['ROBOSIMS_CLIENT_TOKEN'] = self.server.client_token

        # print("Viewer: http://%s:%s/viewer" % (host, port))



        # copy configs as environment variables
        for k, v in config.items():
            env[k] = v

        # show current configs
        robosims.config.pretty_print_config()

        # launch simulator
        if start_unity:
            linux = platform.system() == 'Linux'

            if linux:
                wc = self.setup_window_children(env['DISPLAY'])
                pre_children = wc()
                
            proc = subprocess.Popen(self.unity_command(config), env=env)
            self.pid = proc.pid

            print("launched pid %s" % self.pid)
            if linux:
                atexit.register(lambda: self.kill_self())
                self.server.xwindow_id = self.find_xwindow_id(self.pid, wc, pre_children)
                self.server.window_children = wc

        self.server.start()

    def kill_self(self):
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError as e:
            pass

    def start(self, port=0, start_unity=True):
        config = robosims.config.update_config_from_yaml(self.config_file)
        env = os.environ.copy()

        display = None
        if platform.system() == 'Linux':
            display = env['DISPLAY'] = ':' + config['ROBOSIMS_X_DISPLAY']

        self.server = Server(self.request_queue, self.response_queue, port, display=display)
        self.server_thread = threading.Thread(target=self._start_thread, args=(env, config, port, start_unity))
        self.server_thread.daemon = True
        self.server_thread.start()

    def stop(self):
        if self.server.state == 'waiting_for_response':
            self.response_queue.put(0)
        self.server.wsgi_server.shutdown()
        if self.pid != -1:
            os.kill(self.pid, signal.SIGKILL)

class Server(object):

    def __init__(self, request_queue, response_queue, port=0, display=None):

        self.sct = None

        if display:
            self.sct = mss.linux.MSS(display=display.encode('ascii'))

        app = Flask(__name__, template_folder=os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')))
        self.image_buffer = None 

        self.app = app
        self.client_token = str(uuid.uuid4())
        self.state = 'waiting_for_request'
        self.subscriptions = []
        self.image_queue = deque(maxlen=100)
        self.app.config.update(PROPAGATE_EXCEPTIONS=True, JSONIFY_PRETTYPRINT_REGULAR=False)
        self.port = port
        self.last_rate_timestamp = time.time()
        self.frame_counter = 0
        self.debug_frames_per_interval = 50
        self.xwindow_id = None
        self.wsgi_server = werkzeug.serving.BaseWSGIServer('127.0.0.1', self.port, self.app)

        @app.route('/ping', methods=['get'])
        def ping():
            return 'pong'

        @app.route('/last_image', methods=['get'])
        def last_image():
            return send_file(io.BytesIO(self.image_queue[-1]), 'image/png')

        @app.route('/train', methods=['post'])
        def train():

            if self.client_token:
                token = request.form['token']
                if token is None or token != self.client_token:
                    abort(403)

            if self.frame_counter % self.debug_frames_per_interval == 0:
                now = time.time()
                rate = self.debug_frames_per_interval / float(now - self.last_rate_timestamp)
                print("%s %s/s" % (datetime.datetime.now().isoformat(), rate))
                self.last_rate_timestamp = now

            metadata = json.loads(request.form['metadata'])

            if 'image' in request.files:
                image = request.files['image']
                image_data = image.read()
                self.image_queue.append(image_data)
                image = np.asarray(Image.open(io.BytesIO(image_data)))  # decode image from string encoding

            if 'image_depth' in request.files:
                image_depth = request.files['image_depth']
                image_depth_data = image_depth.read()
                image_depth = np.asarray(Image.open(io.BytesIO(image_depth_data)))  # decode image from string encoding
            else:
                image_depth = None

            if 'image_flow' in request.files:
                image_flow = request.files['image_flow']
                image_flow = image_flow.read()
                image_flow = np.asarray(Image.open(io.BytesIO(image_flow)))  # decode image from string encoding
            else:
                image_flow = None

            event = Event(metadata, self.frame_counter, image, image_depth, image_flow)

            if self.frame_counter != 0:
                # The first event is the result of initializing
                # unity.... it is not the result of any action
                request_queue.put(event)

            self.frame_counter += 1
            self.state = 'waiting_for_response'

            next_action = response_queue.get()
            self.state = 'waiting_for_request'

            #def notify():
            #    for sub in self.subscriptions[:]:
            #        sub.put(filename)

            #gevent.spawn(notify)

            return make_response(json.dumps(next_action))

        @app.route("/subscribe")
        def subscribe():
            def gen():
                q = Queue()
                self.subscriptions.append(q)
                try:
                    while True:
                        data = q.get()
                        yield format_sse(data=data)
                except GeneratorExit:  # Or maybe use flask signals
                    self.subscriptions.remove(q)

            return Response(gen(), mimetype="text/event-stream")

        @app.route('/viewer', methods=['GET'])
        def viewer():
            return render_template('viewer.html')

    def start(self):
        self.wsgi_server.serve_forever()
