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
from a3c.train import train as a3c_train
from learn_distance.train import train as distance_network_train
from learn_translation.train import train as translation_network_train
from learn_class.train import train as class_network_train
from gen_dataset.gen_dataset import gen_dataset
from util.bfs import run_bfs

S3_BUCKET='ai2-vision-robosims'
ENV="Bedroom_04"
BUILDPATH="projects/Bedroom04/builds"

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

    filename = BUILDPATH + '/%s' % build_name
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # if not os.path.isfile(filename):
    #    s3 = boto3.resource('s3')
    #    s3.meta.client.download_file(S3_BUCKET, 'builds/%s' % build_name, filename)

    zipf = zipfile.ZipFile(filename, 'r')
    zipf.extractall(path=BUILDPATH)


@task
def push_linux_build(context):

    build_name = ENV + "-Linux64-%s.zip" % (datetime.datetime.now().isoformat(),)
    archive_name = BUILDPATH + "/%s" % build_name

    zipf = zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED)

    data_path = BUILDPATH + "/" + ENV + '-Linux64_Data'
    exec_path = BUILDPATH + "/" + ENV + '-Linux64'

    for root, dirs, files in os.walk(data_path):
        for f in files:
            fn = os.path.join(root, f)
            print(fn)
            arcname = os.path.relpath(fn, BUILDPATH)
            zipf.write(fn, arcname)

    print(exec_path)
    zipf.write(exec_path, ENV + '-Linux64')
    zipf.close()

    context.run("scp %s amnonh@baymax.cs.washington.edu:builds" % archive_name)

    # s3 = boto3.resource('s3')
    # key = 'builds/%s' % (build_name,)

    # s3.Object(S3_BUCKET, key).put(Body=open(archive_name, 'rb'))
    # print("pushed build %s to %s" % (S3_BUCKET, build_name))


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
        bus_id = 'PCI:' + ':'.join(map(lambda x: str(int(x, 16)), re.split(r'[:\.]', device['Slot'])))
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))
    
    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    return "\n".join(xorg_conf)

@task
def startx(context, display=15):
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")
    records = list(filter(lambda r: r.get('Vendor', '') == 'NVIDIA Corporation' and (r['Class'] == 'VGA compatible controller' or r['Class'] == '3D controller'), pci_records()))

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
def bfs(context, port=0, start_unity=True):
    run_bfs(['--server-config=configs/cfg_bedroom04_drone.yaml'])

@task
def train_a3c(context, port=0, start_unity=True):
    sys.path.append("./networks")
    a3c_train(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=a3c/train.yaml', '--partial-load-model=model_translation/model-1500.cptk' ])

@task
def train_distance(context, port=0, start_unity=True, config="learn_distance/train.yaml"):
    sys.path.append("./networks")
    args = ['--server-config=configs/cfg_bedroom04_drone.yaml']
    args.append("--config=" + config)
    print(args)
    distance_network_train(args)

@task
def test_distance(context, port=0, start_unity=True, config="learn_distance/test.yaml"):
    sys.path.append("./networks")
    args = ['--server-config=configs/cfg_bedroom04_drone.yaml', '--load-model', '--test-only']
    args.append("--config=" + config)
    distance_network_train(args)

@task
def train_translation(context, port=0, start_unity=True, config="learn_translation/train.yaml"):
    sys.path.append("./networks")
    args = ['--server-config=configs/cfg_bedroom04_drone.yaml']
    args.append("--config=" + config)
    translation_network_train(args)

@task
def test_translation(context, port=0, start_unity=True, config="learn_translation/test.html"):
    sys.path.append("./networks")
    args = ['--server-config=configs/cfg_bedroom04_drone.yaml', '--load-model', '--test-only']
    args.append("--config=" + config)
    translation_network_train(args)

@task
def train_class(context, port=0, start_unity=True, config="learn_distance/train.yaml"):
    sys.path.append("./networks")
    args = ['--server-config=configs/cfg_bedroom04_drone.yaml']
    args.append("--config=" + config)
    print(args)
    class_network_train(args)

@task
def test_class(context, port=0, start_unity=True, config="learn_distance/test.yaml"):
    sys.path.append("./networks")
    args = ['--server-config=configs/cfg_bedroom04_drone.yaml', '--load-model', '--test-only']
    args.append("--config=" + config)
    class_network_train(args)

@task
def gen_train_dataset(context, port=0, start_unity=True, config=""):
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config='+config])
    
@task
def gen_train_datasets(context, port=0, start_unity=True):
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/50cm.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/20cm.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/1m.yaml'])

@task
def gen_test_datasets(context, port=0, start_unity=True):
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/test_50cm.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/test_20cm.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/test_1m.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/verify_50cm.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/verify_20cm.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/verify_1m.yaml'])

@task
def gen_rot_datasets(context, port=0, start_unity=True):
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/50cm_rot20.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/20cm_rot20.yaml'])
    gen_dataset(['--server-config=configs/cfg_bedroom04_drone.yaml', '--config=gen_dataset/1m_rot20.yaml'])
