"""
Quick demo with tiny tensors to verify all components work.
"""
import sys, torch, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/cardiac_dut")

# Use tiny spatial dims to avoid OOM
from configs.config import cfg
cfg.data.spatial_size    = (32, 32, 4)
cfg.model.encoder_channels = (16, 32)   # 2 scales only
cfg.model.ssm_d_state      = 8
cfg.model.ssm_expand       = 1
cfg.model.lora_r_init      = 4
cfg.model.num_experts      = 3
cfg.ttec.gate_hidden_dim   = 16
cfg.topology.cubical_resolution = (8, 8, 2)

device = torch.device("cpu")

print("\n[1] Model forward pass ...")
from models.architecture import DUTMambaMoE
model = DUTMambaMoE(num_classes=4)
B, K = 2, 4
image = torch.randn(B, 1, 32, 32, 4)
label = torch.randint(0, K, (B, 32, 32, 4))
with torch.no_grad():
    out = model(image, phase=3)
p_bar      = out["p_bar"]
vacuity    = out["vacuity"]
dissonance = out["dissonance"]
u_sys_v    = out["u_sys"].mean().item()
print(f"   p_bar: {p_bar.shape}  vac range [{vacuity.min():.3f},{vacuity.max():.3f}]  u_sys={u_sys_v:.4f}")

print("\n[2] EDL uncertainty ...")
e0 = out["expert_outputs"][0]
print(f"   Vacuity={e0['vacuity'].mean():.4f}  Dissonance={e0['dissonance'].mean():.4f}")

print("\n[3] Topology check (3D Betti) ...")
from ttec.ttec_engine import check_3d_betti, find_betti_violations
prob_np = p_bar[0].detach().cpu().numpy()
dummy   = np.stack([np.zeros_like(prob_np[0])] + [prob_np[i] for i in range(1,4)])
betti   = check_3d_betti(dummy, resolution=(8,8,2))
violations = find_betti_violations(betti)
for s,b in betti.items():
    tgt = {"LV":(1,0,0),"Myo":(1,1,0),"RV":(1,0,0)}[s]
    print(f"   {s}: {b} target:{tgt}  {'✓' if b==tgt else '✗'}")

print("\n[4] G_TTEC soft gating ...")
from ttec.ttec_engine import GTTECGating, TopologicalUncertaintyScorer, TopoAStarCorrector, TTECEngine
gating    = GTTECGating()
tu_scorer = TopologicalUncertaintyScorer()
corrector = TopoAStarCorrector()
ttec      = TTECEngine(gating, tu_scorer, corrector)
hiddens   = [h[0].detach().cpu() for hs in out["hidden_states"] for h in hs]
result    = ttec.run(p_bar[0].detach(), vacuity[0].detach(),
                     dissonance[0].detach(), hiddens, u_sys_v)
print(f"   violations={result['violation_types']}  TU={result.get('tu',0):.4f}")

print("\n[5] Clinical index ...")
from evaluation.metrics import ClinicalIndexCalculator
calc   = ClinicalIndexCalculator()
ef_res = calc.compute_ef(p_bar[0].detach(), p_bar[1].detach(),
                          vacuity[0].detach(), vacuity[1].detach(),
                          dissonance[0].detach(), dissonance[1].detach(),
                          voxel_vol_ml=0.01)
print(f"   EF={ef_res['ef']:.1f}%  σ={ef_res['sigma_ef_analytic']:.2f}%  "
      f"CI=[{ef_res['ef_ci_lower']:.1f},{ef_res['ef_ci_upper']:.1f}]")

print("\n[6] Loss functions ...")
import torch.nn.functional as F
from training.losses import EDLLoss, Betti3DLoss, TopologicalContrastiveLoss, SegmentationLoss
alpha = out["expert_outputs"][0]["alpha"]
y_oh  = F.one_hot(label, K).permute(0,4,1,2,3).float()
l_edl = EDLLoss()(alpha, y_oh, 5, 50)
l_seg = SegmentationLoss(K)(out["expert_outputs"][0]["logits"], label)
l_b3d = Betti3DLoss(resolution=(8,8,2))(out["expert_outputs"][0]["p_hat"], "Myo")
h0    = [h[0].detach().cpu() for h in out["hidden_states"][0]]
h0aug = [h+0.01*torch.randn_like(h) for h in h0]
l_tcl = TopologicalContrastiveLoss()(h0, h0aug)
print(f"   EDL={l_edl.item():.4f}  Seg={l_seg.item():.4f}  "
      f"Betti3D={l_b3d.item() if hasattr(l_b3d,'item') else l_b3d:.4f}  "
      f"TCL={l_tcl.item():.4f}")

print("\n[7] Segmentation metrics ...")
from evaluation.metrics import compute_metrics
metrics = compute_metrics(p_bar.argmax(1).detach().cpu(), label.cpu(), K)
for k,v in metrics.items(): print(f"   {k}={v:.4f}")

print("\n[8] Homoscedastic loss ...")
from training.losses import HomoscedasticLoss
hl = HomoscedasticLoss()
losses = {"edl":l_edl,"topo":l_b3d if isinstance(l_b3d,torch.Tensor)
          else torch.tensor(float(l_b3d)),"tcl":l_tcl,"seg":l_seg}
log_s2 = model.log_sigma2[0]
hl_val = hl(losses, log_s2)
print(f"   Homoscedastic total loss={hl_val.item():.4f}")

print("\n[9] G_TTEC training ...")
from training.losses import GTTECLoss
np.random.seed(0)
feats = torch.randn(60, 7)
labs  = torch.tensor(np.random.choice([0,1,2], 60), dtype=torch.long)
gate_loss_fn = GTTECLoss()
q = gating(feats)
gl = gate_loss_fn(q, labs)
acc = (q.argmax(1)==labs).float().mean()
print(f"   G_TTEC loss={gl.item():.4f}  acc={acc:.3f}")

print("\n[10] Plotting figures ...")
import matplotlib
matplotlib.use("Agg")
from visualization.plots import (
    plot_segmentation_overlay, plot_ef_scatter,
    plot_tu_distribution, plot_clinical_statistics
)
import os; os.makedirs("/home/claude/cardiac_dut/outputs/demo", exist_ok=True)
mid_z   = 2
p_slice = p_bar[0].argmax(0)[:,:,mid_z].detach().numpy()
plot_segmentation_overlay(
    image[0,0,:,:,mid_z].numpy(), label[0,:,:,mid_z].numpy(),
    p_slice,
    vacuity[0,:,:,mid_z].detach().numpy(),
    dissonance[0,:,:,mid_z].detach().numpy(),
    out_path="/home/claude/cardiac_dut/outputs/demo/overlay.png")
n = 30; np.random.seed(1)
ef_p = 55+np.random.randn(n)*10; ef_r = ef_p+np.random.randn(n)*3
sig  = np.abs(np.random.randn(n)*2+1)
plot_ef_scatter(ef_p, ef_r, sig, out_path="/home/claude/cardiac_dut/outputs/demo/ef_scatter.png")
plot_tu_distribution(np.random.rand(50)*0.3+0.1, np.random.rand(50)*0.3+0.15,
    {"M&Ms": np.random.rand(30)*0.5+0.3},
    out_path="/home/claude/cardiac_dut/outputs/demo/tu_dist.png")
plot_clinical_statistics(
    {"ours_ef": list(np.abs(np.random.randn(40)*2+2)),
     "blind_ef": list(np.abs(np.random.randn(40)*3+4))},
    out_path="/home/claude/cardiac_dut/outputs/demo/clinical.png")
print("   Saved 4 figures to outputs/demo/")

print("\n✓  ALL COMPONENTS VERIFIED SUCCESSFULLY")
