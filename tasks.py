import sys
import boto3
from invoke import task
import os.path
import robosims.server
from robosims.unity import UnityGame
import subprocess
import shlex
import zipfile
import datetime
import platform
import json
import tempfile
import re
from a3c.train import train as network_train
from learn_distance.train import train as distance_network_train
from learn_direction.train import train as direction_network_train

S3_BUCKET='ai2-vision-robosims'

def pci_records():
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(':')[0]] = value

    return records

@task
def install_assets(context):
    s3 = boto3.resource('s3').meta.client
    asset_files = ['HQ_ResidentialHouse.zip', 'Modern_living_room.zip']
    for key in asset_files:
        print("downloading %s" % key)
        if not os.path.isfile(key):
            s3.download_file(Bucket=S3_BUCKET, Key="Assets/" + key, Filename=key)
        context.run("unzip -o -d unity/Assets %s" % key)

@task
def pull_linux_build(context, build_name):

    filename = 'unity/builds/%s' % build_name
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.isfile(filename):
        s3 = boto3.resource('s3')
        s3.meta.client.download_file(S3_BUCKET, 'builds/%s' % build_name, filename)

    zipf = zipfile.ZipFile(filename, 'r')
    zipf.extractall(path='unity/builds')


@task
def push_linux_build(context):

    build_name = "living-room-Linux64-%s.zip" % (datetime.datetime.now().isoformat(),)
    archive_name = "unity/builds/%s" % build_name

    zipf = zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk('unity/builds/living-room-Linux64_Data'):
        for f in files:
            fn = os.path.join(root, f)
            arcname = os.path.relpath(fn, 'unity/builds')
            zipf.write(fn, arcname)

    zipf.write('unity/builds/living-room-Linux64', 'living-room-Linux64')
    zipf.close()

    s3 = boto3.resource('s3')
    key = 'builds/%s' % (build_name,)

    s3.Object(S3_BUCKET, key).put(Body=open(archive_name, 'rb'))
    print("pushed build %s to %s" % (S3_BUCKET, build_name))


@task
def dummy_client(context):
    import requests
    image_data = None
    with open('893.png', 'rb') as f:
        image_data = f.read()
    while True:
        metadata = json.dumps(dict(foo='bar', position='2234'))
        res = requests.post('http://127.0.0.1:8200/train', files={'image':('frame1.png', image_data)}, data=dict(metadata=metadata))
        print(res.json())


def generate_xorg_conf(devices):
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
"""
    screen_records = []
    for i, device in enumerate(devices):
        bus_id = 'PCI:' + ':'.join(map(lambda x: str(int(x)), re.split(r'[:\.]', device['Slot'])))
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))
    
    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    return "\n".join(xorg_conf)

@task
def startx(context, display=5):
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")
    records = list(filter(lambda r: r.get('Vendor', '') == 'NVIDIA Corporation' and r['Class'] == 'VGA compatible controller', pci_records()))

    if not records:
        raise Exception("no nvidia cards found")

    try:
        fd, path = tempfile.mkstemp()
        with open(path, "w") as f:
            f.write(generate_xorg_conf(records))
        command = shlex.split("sudo Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%s" % (path, display))
        subprocess.call(command)
    finally: 
        os.close(fd)
        os.unlink(path)
@task
def build(context, osxintel64=False, linux64=False):

    build_arches = []
    if osxintel64:
        build_arches.append('OSXIntel64')
    if linux64:
        build_arches.append('Linux64')

    for target_arch in build_arches:
        context.run("/Applications/Unity/Unity.app/Contents/MacOS/Unity -quit -batchmode -logFile /dev/stdout -executeMethod Build.%s" % target_arch)


@task
def launch(context, port=0, start_unity=True):
    # cfg_file = 'configs/cfg_bedroom04_navigation.yaml'
    cfg_file = 'configs/cfg_bedroom04_drone.yaml'
    #cfg_file = 'configs/cfg_livingroom_navigation.yaml'
    env = UnityGame(cfg_file, port, start_unity)

    # Do 10 trials
    for i in range(10):
        env.new_episode()

    env.stop()

@task
def train(context, port=0, start_unity=True):
    network_train(['--num_workers=1', '--config=configs/cfg_bedroom04_drone.yaml', '--initialize-weights=posenet.npy'])

@task
def quick_train(context, port=0, start_unity=True):
    network_train(['--num_workers=1', '--config=configs/cfg_bedroom04_drone.yaml'])

@task
def quick_discrete_train(context, port=0, start_unity=True):
    network_train(['--num_workers=1', '--config=configs/cfg_bedroom04_drone.yaml', '--discrete-actions'])

@task
def discrete_train(context, port=0, start_unity=True):
    network_train(['--num_workers=1', '--config=configs/cfg_bedroom04_drone.yaml', '--discrete-actions', '--initialize-weights=posenet.npy'])

@task
def train_distance(context, base_class="GoogleNet", port=0, start_unity=True):
    sys.path.append("./networks")
    print("training {}".format(base_class))
    distance_network_train(['--config=configs/cfg_bedroom04_drone.yaml', 
    '--load-base-weights',
    '--max-distance-delta=0.1', '--max-rotation-delta=0', '--base-class='+ base_class])

@task
def train_direction(context, base_class="GoogleNet", port=0, start_unity=True):
    sys.path.append("./networks")
    print("training {}".format(base_class))
    direction_network_train(['--config=configs/cfg_bedroom04_drone.yaml', '--load-base-weights', '--max-distance-delta=0.1', '--max-rotation-delta=3', '--base-class='+ base_class])
