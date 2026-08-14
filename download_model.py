"""下载 YOLOv3-tiny 模型文件到 model/ 目录"""

import os
import subprocess
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

FILES_LIGHT = {
    "yolov3-tiny.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3-tiny.weights"


def download_light(url, dest):
    print(f"下载 {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  完成: {dest}")


def download_weights(dest):
    print("下载 yolov3-tiny.weights (约 34MB，请耐心等待)...")
    subprocess.run(
        [
            "powershell", "-Command",
            "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
            f"Invoke-WebRequest -Uri '{WEIGHTS_URL}' -OutFile '{dest}'"
        ],
        check=True,
    )
    print(f"  完成: {dest}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, url in FILES_LIGHT.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"已存在，跳过: {filename}")
            continue
        download_light(url, dest)

    weights_path = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
    if os.path.exists(weights_path):
        print("已存在，跳过: yolov3-tiny.weights")
    else:
        download_weights(weights_path)

    print("\n所有模型文件已就绪。")


if __name__ == "__main__":
    main()
