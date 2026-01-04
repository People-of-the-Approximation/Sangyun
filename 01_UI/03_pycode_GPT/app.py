# app.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import ui
import verify
import numpy as np

app = FastAPI()

DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 115200

FIX_LAYER = 4
FIX_HEAD = 3
FIX_MAX_LEN = 768
GEN_TOKENS = 24


@app.get("/attention_ui", response_class=HTMLResponse)
def attention_ui():
    return HTMLResponse(ui.render_attention_ui(DEFAULT_PORT))


@app.post("/attention_generate", response_class=HTMLResponse)
def attention_generate(
    text: str = Form(...),
    port: str = Form(DEFAULT_PORT),
):
    sw_text = ""
    hw_text = ""
    heatmap_b64 = ""
    error_hw = None

    # SW 생성은 항상
    try:
        sw_text = verify.generate_sw(text, max_len=FIX_MAX_LEN, gen_tokens=GEN_TOKENS)
    except Exception as e:
        sw_text = f"(SW generate failed) {e}"

    # HW 생성 (+ fallback)
    try:
        hw_text, hw_status = verify.generate_hw(
            text,
            max_len=FIX_MAX_LEN,
            gen_tokens=GEN_TOKENS,
            port=port,
            baud=DEFAULT_BAUD,
        )
        if hw_status:
            error_hw = hw_status
    except Exception as e:
        hw_text = f"(HW generate failed) {e}"
        error_hw = str(e)

    # heatmap (+ fallback)
    try:
        tokens, attn, hm_status = verify.compute_hw_heatmap(
            text,
            layer=FIX_LAYER,
            head=FIX_HEAD,
            max_len=FIX_MAX_LEN,
            port=port,
            baud=DEFAULT_BAUD,
        )
        heatmap_b64 = ui._attn_to_png_base64(attn)

        if hm_status:
            error_hw = (error_hw + " | " if error_hw else "") + hm_status

    except Exception as e:
        error_hw = (error_hw + " | " if error_hw else "") + f"Heatmap failed: {e}"
        heatmap_b64 = ui._attn_to_png_base64(attn=np.zeros((2, 2), dtype=float))

    return HTMLResponse(
        ui.render_result_page(
            input_text=text,
            sw_text=sw_text,
            hw_text=hw_text,
            heatmap_png_b64=heatmap_b64,
            error_hw=error_hw,
        )
    )
