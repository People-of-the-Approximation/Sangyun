# verify.py
import numpy as np
import torch

import UART_base
from transformers import AutoTokenizer, AutoModelForCausalLM

from VerificationGPT import (
    GPT2AttentionSoftmaxApprox,
    replace_gpt2_attention,
    set_serial_to_model,
    get_last_attention_matrix,
    set_force_store_attn_to_model,
)

DEVICE = "cpu"
MODEL_NAME = "gpt2"

DEFAULT_LAYER = 0
DEFAULT_HEAD = 0
DEFAULT_MAX_LEN = 768
DEFAULT_GEN_TOKENS = 32  # 생성 느낌 좀 더 나게 24 -> 32 추천

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

baseline_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
approx_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
replace_gpt2_attention(approx_model, GPT2AttentionSoftmaxApprox)


# ✅ 문장 생성 목적: 대화형 프롬프트
def build_prompt(user_text: str) -> str:
    return f"User: {user_text}\nAssistant:"


def _tokenize(prompt: str, max_len: int):
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )


def _open_serial_or_none(port: str, baud: int):
    try:
        ser = UART_base.open_serial(port, int(baud), timeout=1.0)
        return ser, None
    except Exception as e:
        return None, f"HW disconnected -> fallback SW-only approx ({e})"


def _postprocess_generated(text: str) -> str:
    """
    UI에서 보기 좋게:
    - 앞뒤 공백 제거
    - 첫 줄만 사용 (프롬프트 복사/불필요한 줄바꿈 방지)
    """
    text = (text or "").strip()
    if not text:
        return ""
    # 첫 줄만
    return text.splitlines()[0].strip()


# ✅ 공통 generate 설정 (SW/HW 동일하게)
def _generate(model, inputs, gen_tokens: int):
    return model.generate(
        input_ids=inputs["input_ids"].to(DEVICE),
        attention_mask=(
            inputs.get("attention_mask", None).to(DEVICE)
            if inputs.get("attention_mask", None) is not None
            else None
        ),
        max_new_tokens=int(gen_tokens),
        do_sample=True,  # ✅ 생성 느낌
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.15,  # ✅ 반복 억제
        no_repeat_ngram_size=3,  # ✅ n-gram 반복 방지
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def generate_sw(
    text: str, *, max_len: int = DEFAULT_MAX_LEN, gen_tokens: int = DEFAULT_GEN_TOKENS
) -> str:
    prompt = build_prompt(text)
    inputs = _tokenize(prompt, max_len=max_len)
    in_len = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        out_ids = _generate(baseline_model, inputs, gen_tokens)

    gen_part = out_ids[0, in_len:]
    out_text = tokenizer.decode(gen_part, skip_special_tokens=True)
    return _postprocess_generated(out_text)


def generate_hw(text: str, *, max_len: int, gen_tokens: int, port: str, baud: int):
    """
    반환: (generated_text, status_message_or_None)
    - HW 연결되면 row0 softmax는 UART로 처리
    - HW 미연결이면 SW-only approx로 계속 진행
    """
    prompt = build_prompt(text)
    inputs = _tokenize(prompt, max_len=max_len)
    in_len = int(inputs["input_ids"].shape[1])

    ser, status = _open_serial_or_none(port, baud)

    try:
        set_serial_to_model(approx_model, ser)

        with torch.no_grad():
            out_ids = _generate(approx_model, inputs, gen_tokens)

        gen_part = out_ids[0, in_len:]
        out_text = tokenizer.decode(gen_part, skip_special_tokens=True)
        return _postprocess_generated(out_text), status

    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        set_serial_to_model(approx_model, None)


def compute_hw_heatmap(
    text: str,
    *,
    layer: int = DEFAULT_LAYER,
    head: int = DEFAULT_HEAD,
    max_len: int = DEFAULT_MAX_LEN,
    port: str,
    baud: int,
):
    """
    반환: (tokens, attn_matrix, status_message_or_None)
    - heatmap은 forward 1회로 last_attn 저장 후 꺼냄
    """
    prompt = build_prompt(text)
    inputs = _tokenize(prompt, max_len=max_len)

    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    ser, status = _open_serial_or_none(port, baud)

    try:
        # ✅ last_attn 강제 저장
        set_force_store_attn_to_model(approx_model, True)

        set_serial_to_model(approx_model, ser)

        with torch.no_grad():
            _ = approx_model(
                input_ids=inputs["input_ids"].to(DEVICE),
                attention_mask=(
                    inputs.get("attention_mask", None).to(DEVICE)
                    if inputs.get("attention_mask", None) is not None
                    else None
                ),
                output_attentions=False,
                use_cache=False,
            )

        attn = np.asarray(
            get_last_attention_matrix(approx_model, layer=int(layer), head=int(head)),
            dtype=np.float64,
        )
        return tokens, attn, status

    finally:
        set_force_store_attn_to_model(approx_model, False)

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        set_serial_to_model(approx_model, None)
