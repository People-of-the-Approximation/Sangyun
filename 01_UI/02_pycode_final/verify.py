# verify.py
import numpy as np
import torch

import UART_base
from VerificationBERT import (
    build_models_sst2,
    set_serial_to_model,
    get_last_attention_matrix,
)

# =========================
# Settings
# =========================
DEVICE = "cpu"  # 필요하면 "cuda"로 변경
LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

# Load models once (import 시 1번만)
tokenizer, baseline_model, approx_model = build_models_sst2(device=DEVICE)


# =========================
# Utils
# =========================
def predict_from_logits(logits_2):
    probs = torch.softmax(logits_2, dim=-1).cpu().numpy()
    pred = int(torch.argmax(logits_2).item())
    return {
        "pred_id": pred,
        "pred_label": LABEL[pred],
        "p_neg": float(probs[0]),
        "p_pos": float(probs[1]),
    }


def predict_sw_only(text: str, *, max_len: int):
    """SW(baseline)로 분류만 수행 (attention은 안 뽑음)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    with torch.no_grad():
        out = baseline_model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=(
                inputs.get("attention_mask", None).to(DEVICE)
                if inputs.get("attention_mask", None) is not None
                else None
            ),
            output_attentions=False,
        )
    logits = out.logits[0].detach()
    return predict_from_logits(logits)


def compute_sw_all(text: str, *, layer: int, head: int, max_len: int):
    """SW(baseline)로 (분류 + attention matrix)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    with torch.no_grad():
        out = baseline_model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=(
                inputs.get("attention_mask", None).to(DEVICE)
                if inputs.get("attention_mask", None) is not None
                else None
            ),
            output_attentions=True,
        )

    # prediction
    pred = predict_from_logits(out.logits[0].detach())

    # attention
    if out.attentions is None:
        raise RuntimeError(
            "SW model did not return attentions (output_attentions=True failed)."
        )

    L = len(out.attentions)
    layer = max(0, min(int(layer), L - 1))

    attn_l = out.attentions[layer]  # (B, heads, T, T)
    H = int(attn_l.shape[1])
    head = max(0, min(int(head), H - 1))

    attn = attn_l[0, head].detach().cpu().numpy().astype(np.float64)

    return tokens, attn, pred


def compute_hw_all(
    text: str, *, layer: int, head: int, max_len: int, port: str, baud: int
):
    """
    HW(approx)로 (분류 + attention matrix)
    - UART 열고 approx_model에 serial 주입
    - output_attentions=True로 forward → logits + last_attn 확보
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    ser = UART_base.open_serial(port, int(baud), timeout=1.0)
    try:
        set_serial_to_model(approx_model, ser)
        with torch.no_grad():
            out = approx_model(
                input_ids=inputs["input_ids"].to(DEVICE),
                attention_mask=(
                    inputs.get("attention_mask", None).to(DEVICE)
                    if inputs.get("attention_mask", None) is not None
                    else None
                ),
                output_attentions=True,
            )

        pred = predict_from_logits(out.logits[0].detach())

        attn = get_last_attention_matrix(approx_model, layer=int(layer), head=int(head))
        attn = np.asarray(attn, dtype=np.float64)

        return tokens, attn, pred
    finally:
        try:
            ser.close()
        except Exception:
            pass
