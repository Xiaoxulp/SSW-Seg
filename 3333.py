import torch
import warnings

from network.model import ConDSeg
from network.model_stage1 import ConDSegStage1
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
warnings.filterwarnings("ignore")



# =========================
# Stage-1 Complexity
# =========================
def compute_stage1_complexity():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConDSegStage1(H=128, W=128).to(device)
    model.eval()

    dummy = torch.randn(1, 3, 128, 128).to(device)

    # ---- Params & FLOPs ----
    flops, params = profile(model, inputs=(dummy,), verbose=False)

    print("\n====== Stage-1 (ConDSegStage1) ======")
    print(f"Params : {params / 1e6:.2f} M")
    print(f"FLOPs  : {flops / 1e9:.2f} G")

    # ---- GPU Memory (Inference) ----
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory (Inference): {mem:.2f} GB")
    else:
        print("GPU Memory: skipped (CPU mode)")


# =========================
# Stage-2 Complexity
# =========================
def compute_stage2_complexity():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConDSeg(H=128, W=128).to(device)
    model.eval()

    dummy = torch.randn(1, 3, 128, 128).to(device)

    # ---- Params & FLOPs ----
    flops, params = profile(model, inputs=(dummy,), verbose=False)

    print("\n====== Stage-2 (ConDSeg) ======")
    print(f"Params : {params / 1e6:.2f} M")
    print(f"FLOPs  : {flops / 1e9:.2f} G")

    # ---- GPU Memory (Inference) ----
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory (Inference): {mem:.2f} GB")
    else:
        print("GPU Memory: skipped (CPU mode)")


# =========================
# (Optional) FVCore FLOPs
# =========================
def compute_fvcore_flops():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConDSeg(H=128, W=128).to(device)
    model.eval()

    dummy = torch.randn(1, 3, 128, 128).to(device)

    flops = FlopCountAnalysis(model, dummy)
    print("\n====== FVCore FLOPs ======")
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} G")


# =========================
# (Optional) Param Table
# =========================
def print_param_table():
    model = ConDSeg(H=128, W=128)
    print(parameter_count_table(model))


# =========================
# Main
# =========================
if __name__ == "__main__":
    compute_stage1_complexity()
    compute_stage2_complexity()

    # 下面两个需要时再打开
    # compute_fvcore_flops()
    # print_param_table()
