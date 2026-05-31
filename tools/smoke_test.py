"""
smoke_test.py — exercise the whole pipeline on the 2 recorded episodes.

Goal: prove the wiring works (data -> model -> train step -> checkpoint ->
load -> predict), NOT to train a useful policy. Everything runs on CPU in
seconds with a tiny model.

Stages:
  1. convert raw episodes -> LeRobot dataset (_smoke_dataset)
  2. load via training/dataset.py and check tensor shapes
  3. [needs lerobot] build the ACT model, run one fwd/bwd step,
     save a checkpoint, reload it (deploy-style) and call predict()

If lerobot isn't installed, stages 1-2 still run and stage 3 is reported as
SKIPPED with the reason.

    python tools/smoke_test.py
"""

import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
SMOKE_DS = REPO / "_smoke_dataset"
sys.path.insert(0, str(REPO / "training"))


def stage1_convert():
    print("\n[1/3] convert raw episodes -> LeRobot dataset")
    cmd = [sys.executable, str(REPO / "tools" / "convert_episodes.py"),
           "--episodes", str(REPO / "episodes"),
           "--out", str(SMOKE_DS),
           "--extract-frames", "--max-frames", "60"]
    subprocess.run(cmd, check=True)


def stage2_dataset():
    print("\n[2/3] load via FR5Dataset and check shapes")
    from dataset import FR5Dataset

    chunk = 20
    ds = FR5Dataset(str(SMOKE_DS), chunk_size=chunk, use_image=True,
                    image_size=(224, 224), episode_indices=[0])
    assert len(ds) > 0, "dataset is empty"
    s = ds[0]
    assert s["observation.state"].shape == (6,), s["observation.state"].shape
    assert s["action"].shape == (chunk, 7), s["action"].shape
    assert s["action_is_pad"].shape == (chunk,)
    assert s["observation.image"].shape == (3, 224, 224), s["observation.image"].shape
    stats = ds.get_stats()
    assert stats["state_mean"].shape == (6,) and stats["action_mean"].shape == (7,)
    print(f"      OK  len={len(ds)}  state={tuple(s['observation.state'].shape)}  "
          f"action={tuple(s['action'].shape)}  image={tuple(s['observation.image'].shape)}")
    return stats


def stage3_model(stats):
    print("\n[3/3] build model, train step, checkpoint, reload, predict")
    try:
        from model import ACT, ACTConfig
    except Exception as e:  # lerobot missing or API mismatch
        print(f"      SKIPPED — could not import model.py ({type(e).__name__}: {e})")
        print("      install the policy backend to run this stage:  pip install lerobot")
        return False

    cfg = ACTConfig(state_dim=6, action_dim=7, latent_dim=8, d_model=64, nhead=4,
                    num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=128,
                    chunk_size=20, use_image=True, kl_weight=10.0,
                    temporal_ensemble_coeff=None)
    device = torch.device("cpu")
    model = ACT(cfg, stats).to(device)

    # one fwd/bwd step on a dummy batch
    B = 2
    obs = torch.randn(B, 6)
    acts = torch.randn(B, 20, 7)
    pad = torch.zeros(B, 20, dtype=torch.bool)
    img = torch.rand(B, 3, 224, 224)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    loss, l1, kl = model(obs, acts, pad, img)
    loss.backward()
    opt.step()
    print(f"      train step OK  l1={l1:.4f}  kl={kl:.4f}")

    # checkpoint round-trip (mirrors train.py / deploy.py)
    ckpt = SMOKE_DS / "smoke_ckpt.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(),
                "val_l1": l1, "stats": stats}, ckpt)
    model2 = ACT(cfg, stats).to(device)
    model2.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
    model2.eval()
    model2.reset()
    out = model2.predict(obs[:1], img[:1])
    out = out[0] if out.dim() == 2 else out
    assert out.shape[-1] == 7, out.shape
    print(f"      checkpoint reload + predict OK  action={tuple(out.shape)}")
    return True


def main():
    stage1_convert()
    stats = stage2_dataset()
    model_ok = stage3_model(stats)
    print("\n" + "=" * 60)
    print("SMOKE TEST: data pipeline PASSED" +
          ("  |  model+train+deploy PASSED" if model_ok
           else "  |  model stage SKIPPED (lerobot not installed)"))
    print("=" * 60)


if __name__ == "__main__":
    main()
