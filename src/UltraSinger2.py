import sys
import torch

# ---- performance tweak ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---- inject better defaults if user didn't specify ----
DEFAULT_ARGS = [
    "--whisper", "large-v3",
    "--demucs", "htdemucs_ft",
    "--crepe", "full",
    "--crepe_step_size", "5",
]

def inject_defaults(argv):
    present = set(argv)
    out = list(argv)

    for flag, value in zip(DEFAULT_ARGS[0::2], DEFAULT_ARGS[1::2]):
        if flag not in present:
            out.extend([flag, value])

    return out

# ---- patch argv BEFORE UltraSinger parses ----
sys.argv = [sys.argv[0]] + inject_defaults(sys.argv[1:])

# ---- run UltraSinger ----
import UltraSinger

if __name__ == "__main__":
    UltraSinger.main(sys.argv[1:])

