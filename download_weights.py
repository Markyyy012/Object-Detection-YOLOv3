import os
import sys

import requests

WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"

EXPECTED_WEIGHTS_SIZE = 248007048


def download(url, dest, chunk_size=1 << 15):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  exists, skipping: {dest}")
        return True
    print(f"Downloading {os.path.basename(dest)} ({url})")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    bar = "#" * int(pct / 2)
                    sys.stdout.write(f"\r  [{bar:<50}] {pct:6.2f}%  {done / 1e6:.1f}/{total / 1e6:.1f} MB")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    return True


def main():
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)

    cfg_path = os.path.join(weights_dir, "yolov3.cfg")
    weights_path = os.path.join(weights_dir, "yolov3.weights")

    download(CFG_URL, cfg_path)
    download(WEIGHTS_URL, weights_path)

    size = os.path.getsize(weights_path)
    if size == EXPECTED_WEIGHTS_SIZE:
        print(f"OK: yolov3.weights verified ({size} bytes).")
    else:
        print(
            f"WARNING: yolov3.weights size {size} differs from expected "
            f"{EXPECTED_WEIGHTS_SIZE}; try re-running download_weights.py."
        )


if __name__ == "__main__":
    main()
