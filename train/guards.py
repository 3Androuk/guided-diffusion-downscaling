"""Shared collapse protection for the trainers.

Four of the six 20-var BriCS arms collapsed to the trivial solution mid-run
(jobs 6169725, 6189791, 6206370, 6207062, 6218150, 6226877 -- the last three
were rescued). The mechanism is a rare pathological gradient that AdamW turns
into a sustained ~lr-per-element walk into an absorbing basin; grad_clip does
not stop it because Adam normalizes by its own second moment. See
utils.SpikeGuard for the full account.

train_diffusion.py grew this protection inline. train_directmap.py and
train_residual.py had none, so rather than paste the fiddly part a third and
fourth time, the RESTORE step lives here: reloading weights, EMA, Adam moments
and scaler together is the part that is easy to get subtly wrong, and getting
it wrong (e.g. leaving the diverged optimizer moments in place) silently
reproduces the collapse it is meant to undo.
"""

import torch


def restore_best(path, raw_model, ema, opt, scaler, device):
    """Reload the pre-collapse state from a `_best` checkpoint, in place.

    Adam's moments must come back with the weights. Restoring weights alone
    leaves the second-moment estimates carrying the pathological direction,
    which re-applies it for the next ~1/(1-beta2) steps and walks straight back
    into the same basin.

    Returns the checkpoint's epoch, or None when the file does not exist yet
    (a collapse before the first save leaves nothing to roll back to).
    """
    if not path.exists():
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ck["model"])
    if ema is not None and "ema" in ck:
        ema.load_state_dict(ck["ema"])
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    if scaler is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    return ck.get("epoch")


def spike_skip(spike, grad_norm, scaler, step, is_main):
    """True when this optimizer step should be skipped entirely.

    Feed the PRE-clip gradient norm (what clip_grad_norm_ returns). The caller
    must then call scaler.update() and `continue` WITHOUT scaler.step(), so
    Adam's moments never see the bad direction.

    Under DDP no cross-rank agreement is needed: gradients are all-reduced in
    the backward pass, so every rank computes an identical norm and takes the
    same branch. Do not add a collective here -- a collective inside a branch
    is how ranks deadlock.
    """
    if spike is None or not spike.check(float(grad_norm)):
        return False
    if is_main and spike.skipped <= 20:
        print(f"  spike guard: skipped step {step} — {spike.last_reason}", flush=True)
    return True


def spike_recover(spike, is_main, step):
    """Break a spike-guard deadlock. Call right after a skipped step.

    Returns True when the guard was re-calibrated. Without this a run can accept
    ZERO optimizer steps for the rest of its epoch budget while every metric
    looks merely 'converged' -- measured on jobs 6238082 and 6362657, which
    trained for 26 and 33 epochs of a 200-epoch budget.
    """
    if spike is None or not spike.exhausted():
        return False
    if is_main:
        print(f"  spike guard: {spike.consecutive} consecutive skips at step {step} "
              f"— the running median is stale, re-calibrating (was not spiking, "
              f"the model moved to a higher-gradient region)", flush=True)
    spike.reset(spike.scale())   # re-arm at the healthy scale, never unarmed
    return True
