# realtime_pedalnet.py
import argparse, threading, queue, time, sys
import numpy as np
import sounddevice as sd
import torch

from model import PedalNet                           # your network

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)        # .ckpt
parser.add_argument("--sample-rate", type=int, default=48000)
parser.add_argument("--block", type=int, default=256, help="IO buffer size (frames)")
parser.add_argument("--device-in", type=int, default=None)
parser.add_argument("--device-out", type=int, default=None)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

# ---------- Model ----------
ckpt   = torch.load(args.model, map_location="cpu")
hp     = ckpt["hyper_parameters"]
net    = PedalNet(hp)
net.load_state_dict(ckpt["state_dict"])
net.eval()

device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
net.to(device)

# make sure the network accepts arbitrary lengths (convolutions only)
dummy = torch.randn(1, 1, args.block).to(device)
with torch.no_grad(): net(dummy)


# ---------- Shared state ----------
bypass      = False
latency_buf = queue.Queue(maxsize=200)       # for running average
block_dur   = args.block / args.sample_rate  # I/O buffering latency

# ---------- Helper ----------


# ---------- Key listener ----------
def key_listener():
    global bypass
    print("Press <space> to toggle BYPASS, <q> to quit.")
    while True:
        k = sys.stdin.read(1)
        if k.lower() == "q":
            print("Quitting …")
            sd.stop(); break
        if k == " ":
            bypass = not bypass
            print(f"[BYPASS {'ON' if bypass else 'OFF'}]")

def receptive_field(net, length=32768, device="cpu"):
    """Return the number of samples of left‑context the model really uses."""
    x = torch.zeros(1, 1, length, device=device)   # <-- build on same device
    x[0, 0, length // 2] = 1.0                     # delta pulse in the middle
    with torch.no_grad():
        y = net(x)
    nz = (y.squeeze() != 0).nonzero().squeeze()
    return int(nz[-1] - nz[0] + 1)

R = receptive_field(net, device=device)

ctx = np.zeros(R-1, dtype=np.float32)     # global

threading.Thread(target=key_listener, daemon=True).start()

def running_ms(latq):
    lat = list(latq.queue) or [0.0]
    return sum(lat) / len(lat)

# … keep your existing imports, model loading, receptive_field, ctx, etc.

# inside callback, add timing + stats:
def callback(indata, outdata, frames, t, status):
    global ctx

    if status:
        print(status, file=sys.stderr)

    x = indata[:, 0].copy()

    # ——— inference timing ———
    t0 = time.perf_counter()
    with torch.no_grad():
        inp = np.concatenate((ctx, x))                      # build with context
        y = net(torch.from_numpy(inp)
                   .float()
                   .unsqueeze(0).unsqueeze(0)
                   .to(device))
    infer_time = (time.perf_counter() - t0) * 1e3            # ms

    # ——— update latency buffer ———
    if latency_buf.full():
        latency_buf.get()
    latency_buf.put(infer_time)

    # ——— slice out the block ———
    y = y.squeeze().cpu().numpy()
    out_block = y[-frames:]

    GAIN = 2.0
    outdata[:, 0] = out_block * GAIN
    outdata[:, 1:] = outdata[:, :1]

    # ——— roll context ———
    ctx = inp[-(R-1):]

    # ——— compute running stats ———
    avg_infer = running_ms(latency_buf)                      # ms
    buf_ms    = block_dur * 1e3                              # ms
    total_ms  = avg_infer + buf_ms

    # ——— print once per second ———
    # t.outputBufferDacTime increments by block_dur each block,
    # so modulus 1 s tells us “once per second”
    if (t.outputBufferDacTime % 1) < block_dur:
        print(f"Inference: {infer_time:6.2f} ms  | "
              f"avg infer: {avg_infer:6.2f} ms  | "
              f"buf: {buf_ms:5.1f} ms  | "
              f"total: {total_ms:6.2f} ms   ",
              end="\r")

# … rest of your stream‐opening code unchanged …

# ---------- Open stream ----------
# pick devices (call sd.query_devices() to list indices)
sd.default.samplerate = args.sample_rate
sd.default.blocksize  = args.block
sd.default.dtype      = "float32"
sd.default.latency    = ("low", "low")
# force ASIO on Windows
extra = sd.AsioSettings(channel_selectors=[0, 1]) if sys.platform == "win32" else None


with sd.Stream(channels=(1,2),
               dtype="float32",
               callback=callback,
               device=(args.device_in, args.device_out),
               extra_settings=extra):
    print("Stream started — Ctrl‑C to stop.\n")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
