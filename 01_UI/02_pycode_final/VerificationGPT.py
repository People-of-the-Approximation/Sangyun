# VerificationGPT.py
from __future__ import annotations

from typing import Optional
import numpy as np
import torch

from softmax_packet import softmax_fpga_variable

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
except Exception as e:
    raise RuntimeError(
        "Cannot import GPT2Attention. Check your transformers version."
    ) from e


def softmax_local(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x)
    e = np.exp(x - m)
    s = np.sum(e)
    if s <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return e / s


class GPT2AttentionSoftmaxApprox(GPT2Attention):
    """
    GPT-2 Attention with hybrid softmax:
      - row 0 softmax: HW(UART) if serial is available
      - row 1.. softmax: SW(local)
    Stores last_attn for heatmap.
    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        # transformers 버전마다 init signature가 달라서 안전하게 처리
        try:
            super().__init__(
                config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
            )
        except TypeError:
            super().__init__(config, is_cross_attention=is_cross_attention)

        self.ser = None
        self.last_attn: Optional[np.ndarray] = None

        # ✅ 추가: heatmap 강제 저장 플래그
        self.force_store_attn: bool = False

        # HW softmax 입력에서 mask 효과를 내기 위한 낮은 값
        # (네가 말한 "-32를 0x8000으로 처리" 기준)
        self.pad_value = -32.0

    def set_serial(self, ser):
        self.ser = ser

    # ✅ 추가: 외부에서 last_attn 저장 강제 토글
    def set_force_store_attn(self, flag: bool):
        self.force_store_attn = bool(flag)

    @staticmethod
    def _shape_qkv(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        # (B,T,embed) -> (B,H,T,Dh)
        B, T, _ = x.shape
        return x.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states,
        past_key_value=None,  # ✅ 최신 transformers 인자
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,  # ✅ 버전 차이 흡수
    ):
        layer_past = past_key_value

        # ✅ 기존 want_attn 로직 + force_store_attn 반영
        # (원본은 output_attentions/kwargs만 반영 :contentReference[oaicite:1]{index=1})
        want_attn = bool(output_attentions) or bool(
            kwargs.get("output_attentions", False)
        )
        want_attn = want_attn or bool(getattr(self, "force_store_attn", False))

        # ---- QKV ----
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        query = self._shape_qkv(query, self.num_heads, self.head_dim)
        key = self._shape_qkv(key, self.num_heads, self.head_dim)
        value = self._shape_qkv(value, self.num_heads, self.head_dim)

        # past concat
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present = (key, value) if use_cache else None

        B, H, Tq, Dh = query.shape
        Tk = key.shape[2]

        # attention_mask를 (B,Tk)로 만들어서 masked token의 V=0 처리에 사용
        mask_np = None
        if attention_mask is not None:
            m = attention_mask
            while m.dim() > 2:
                m = m.squeeze(1)
            mask_np = m.detach().cpu().numpy()

        if want_attn:
            self.last_attn = np.zeros((B, H, Tq, Tk), dtype=np.float64)
        else:
            self.last_attn = None

        q_np_all = query.detach().cpu().numpy()
        k_np_all = key.detach().cpu().numpy()
        v_np_all = value.detach().cpu().numpy()

        out = torch.zeros((B, H, Tq, Dh), device=query.device, dtype=query.dtype)

        for b in range(B):
            masked = None
            if mask_np is not None:
                masked = (mask_np[b] == 0) | (mask_np[b] < 0)

            for h in range(H):
                Q = q_np_all[b, h]  # (Tq,Dh)
                K = k_np_all[b, h]  # (Tk,Dh)
                V = v_np_all[b, h]  # (Tk,Dh)

                Vm = V.copy()
                if masked is not None:
                    Vm[masked] = 0.0

                attn_mat = np.zeros((Tq, Tk), dtype=np.float64) if want_attn else None
                out_np = np.zeros((Tq, Dh), dtype=np.float64)

                # causal mask: query index i가 참조할 수 있는 key 최대 인덱스
                past_len = Tk - Tq  # past 포함 시

                for i in range(Tq):
                    scores = (K @ Q[i].T) / np.sqrt(Dh)  # (Tk,)

                    # causal: 미래 토큰은 pad_value로 내려서 softmax에서 거의 0 되게
                    allowed_max = past_len + i
                    if allowed_max + 1 < Tk:
                        scores[allowed_max + 1 :] = self.pad_value

                    # ✅ row0만 HW softmax (UART 연결 시), 그 외 SW local
                    # (원본 그대로 :contentReference[oaicite:2]{index=2})
                    if (i == 0) and (self.ser is not None):
                        probs = softmax_fpga_variable(
                            self.ser,
                            scores,
                            pad_value=self.pad_value,
                            deadline_s=2.0,
                        )
                        probs = np.asarray(probs, dtype=np.float64)
                    else:
                        probs = softmax_local(scores)

                    if want_attn:
                        attn_mat[i, :] = probs

                    out_np[i, :] = probs @ Vm

                out[b, h] = torch.tensor(out_np, device=query.device, dtype=query.dtype)

                if want_attn:
                    self.last_attn[b, h, :, :] = attn_mat

        # merge heads -> (B,T,embed)
        context = out.permute(0, 2, 1, 3).contiguous().view(B, Tq, H * Dh)
        attn_output = self.c_proj(context)
        attn_output = self.resid_dropout(attn_output)

        # attn_weights는 None (heatmap은 last_attn 사용)
        return attn_output, present, None


def replace_gpt2_attention(model: torch.nn.Module, NewAttnClass):
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError("Model is not GPT-2 style (missing transformer.h).")

    for idx, block in enumerate(model.transformer.h):
        old_attn = block.attn
        try:
            new_attn = NewAttnClass(
                model.config, is_cross_attention=False, layer_idx=idx
            )
        except TypeError:
            new_attn = NewAttnClass(model.config, is_cross_attention=False)

        new_attn.load_state_dict(old_attn.state_dict(), strict=True)
        block.attn = new_attn


def set_serial_to_model(model: torch.nn.Module, ser):
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError("Model is not GPT-2 style.")

    for block in model.transformer.h:
        attn = block.attn
        if hasattr(attn, "set_serial"):
            attn.set_serial(ser)


def clear_serial_from_model(model: torch.nn.Module):
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        return
    for block in model.transformer.h:
        attn = block.attn
        if hasattr(attn, "set_serial"):
            attn.set_serial(None)


def get_last_attention_matrix(
    model: torch.nn.Module, layer: int = 0, head: int = 0
) -> np.ndarray:
    blocks = model.transformer.h
    L = len(blocks)
    layer = max(0, min(int(layer), L - 1))

    attn = blocks[layer].attn
    if not hasattr(attn, "last_attn") or attn.last_attn is None:
        raise RuntimeError(
            "No attention stored. Run forward with output_attentions=True first."
        )

    a = attn.last_attn  # (B,H,T,Tk)
    B, H, T, Tk = a.shape
    head = max(0, min(int(head), H - 1))
    return np.asarray(a[0, head], dtype=np.float64)


# ✅ 추가: verify.py에서 쓰는 helper (heatmap 강제 저장 on/off)
def set_force_store_attn_to_model(model: torch.nn.Module, flag: bool):
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        return
    for block in model.transformer.h:
        attn = block.attn
        if hasattr(attn, "set_force_store_attn"):
            attn.set_force_store_attn(flag)
