import torch
from model import PedalNet         

ckpt = torch.load(r"D:\Desktop\Projects\PedalNetRT\models\Fender_deluxe_reverb\Fender_deluxe_reverb.ckpt", map_location="cpu")
hp   = ckpt["hyper_parameters"]

# 1. Build model and load weights
net = PedalNet(hp)
net.load_state_dict(ckpt["state_dict"])
net.eval()

# 2. Grab just the inference part
model = net.wavenet          # or create your own nn.Module that wraps preprocessing + wavenet

# 3. Script it
scripted = torch.jit.script(model)

# 4. (Optional) quick sanity check
x = torch.randn(1, 1, 8)          # batch=1, mono, 2 048‑sample chunk
with torch.no_grad():
    y_py  = model(x)
    y_ts  = scripted(x)
print("max_abs_err:", (y_py - y_ts).abs().max())

# 5. Save
scripted.save("test.pt")
