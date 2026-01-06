# ui_gpt_page.py
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm


def _attn_to_png_base64(attn: np.ndarray) -> str:
    """
    GPT-2 attention heatmap renderer (BERT palette matched)
    - Expect attn shape: (T, T)
    - Attention probs should be in [0, 1]
    - Use SAME 12-step palette as BERT page (Discrete bins via BoundaryNorm)
    """
    attn = np.asarray(attn, dtype=np.float32)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # ✅ BERT와 동일 팔레트 생성(12-step)
    base_colors = [
        "#F6EAE8",
        "#F2CEBE",
        "#ECAE96",
        "#E7916F",
        "#E47950",
        "#E06338",
        "#D85D34",
        "#CB552E",
        "#BD4E2A",
        "#A84121",
    ]

    # 밝은색 부분 보간으로 12단 구성
    light_cmap = LinearSegmentedColormap.from_list("light_part", base_colors[:5])
    light_colors = [mcolors.to_hex(light_cmap(i / 6)) for i in range(7)]  # 7개
    dark_colors = base_colors[5:]  # 5개
    colors_12 = light_colors + dark_colors  # 총 12개

    bounds = np.linspace(0.0, 1.0, len(colors_12) + 1)
    cmap = ListedColormap(colors_12)
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(attn, aspect="auto", cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Attention Heatmap (HW hybrid: row0=HW, others=SW)")
    ax.set_xlabel("Key token index")
    ax.set_ylabel("Query token index")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_attention_ui(default_port: str):
    # (필요하면 나중에 page1으로 통합 가능)
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>GPT Softmax HW Demo</title>
        <link rel="stylesheet" href="/static/style.css" />
        <style>
          html, body {{
            height: 100%;
            margin: 0;
            font-family: 'Satoshi', Arial, sans-serif;
            background: #fff;
            color: #111;
          }}
          .page {{
            width: min(1200px, 94vw);
            margin: 0 auto;
            padding: 24px 0;
          }}
          .title {{
            font-size: 40px;
            font-weight: 500;
            color: #FC5230;
            margin: 0 0 10px 0;
          }}
          .hint {{
            color: #6b7280;
            font-size: 14px;
            line-height: 1.4;
            margin-bottom: 18px;
          }}
          .row {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-top: 10px;
          }}
          input[type=text] {{
            padding: 10px 12px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            font-family: 'Satoshi', Arial, sans-serif;
            font-size: 16px;
          }}
          #text {{
            flex: 1;
          }}
          button {{
            padding: 10px 14px;
            border: none;
            border-radius: 10px;
            background: #FC5230;
            color: #fff;
            font-family: 'Satoshi', Arial, sans-serif;
            font-weight: 500;
            cursor: pointer;
          }}
          button:hover {{
            opacity: 0.92;
          }}
        </style>
      </head>
      <body>
        <div class="page">
          <div class="title">GPT Demo</div>
          <div class="hint">
            Fixed: baud=115200, layer=0, head=0, max_len=768<br/>
            Heatmap: 1 forward. Softmax: <b>row0=HW</b>, others=<b>SW</b>.
          </div>

          <form method="post" action="/attention_generate">
            <div class="row">
              <input id="text" type="text" name="text" placeholder="Enter an English sentence..." />
              <button type="submit">Run</button>
            </div>

            <div class="row">
              <label style="color:#6b7280;">UART port</label>
              <input type="text" name="port" value="{default_port}" style="width:140px;" />
              <input type="hidden" name="model" value="gpt" />
              <input type="hidden" name="mode" value="hw" />
              <input type="hidden" name="layer" value="0" />
              <input type="hidden" name="head" value="0" />
              <input type="hidden" name="max_len" value="768" />
              <input type="hidden" name="baud" value="115200" />
            </div>
          </form>
        </div>
      </body>
    </html>
    """


def render_result_page(
    *,
    input_text: str,
    sw_text: str,
    hw_text: str,
    heatmap_png_b64: str,
    error_hw: str | None = None,
):
    # BERT 결과 페이지처럼: 왼쪽=heatmap, 오른쪽=텍스트 카드(입력/SW/HW)
    err_html = ""
    if error_hw:
        err_html = f"""
        <div class="errbox">
          <div class="errtitle">HW error</div>
          <pre class="errpre">{error_hw}</pre>
        </div>
        """

    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>GPT Result</title>

        <!-- ✅ Satoshi 로드 -->
        <link rel="stylesheet" href="/static/style.css" />

        <style>
          :root {{
            --bg: #ffffff;
            --text: #111;
            --muted: #6b7280;
            --line: #e5e7eb;
          }}

          html, body {{
            height: 100%;
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: 'Satoshi', Arial, sans-serif;
          }}

          .wrap {{
            display: flex;
            height: 100vh;
            width: 100vw;
            padding: 24px 28px;
            gap: 28px;
            box-sizing: border-box;
          }}

          .left {{
            display: flex;
            align-items: center;
            justify-content: center;
          }}

          .heatmap {{
            border: 1px solid var(--line);
            max-height: calc(100vh - 48px);
            max-width: 64vw;
            width: auto;
            height: auto;
            border-radius: 10px;
          }}

          .right {{
            flex: 1 1 auto;
            min-width: 420px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
          }}

          .cards {{
            display: flex;
            flex-direction: column;
            gap: 14px;
          }}

          .card {{
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 14px 16px;
            background: #fff;
          }}

          .label {{
            color: var(--muted);
            font-size: 16px;
            font-weight: 400;
            margin-bottom: 8px;
          }}

          .content {{
            font-size: 18px;
            font-weight: 400;
            white-space: pre-wrap;
            line-height: 1.35;
          }}

          .errbox {{
            border: 1px solid #f5c2c7;
            background: #fff5f5;
            border-radius: 14px;
            padding: 12px 14px;
          }}
          .errtitle {{
            color: #b42318;
            font-weight: 500;
            margin-bottom: 6px;
          }}
          .errpre {{
            margin: 0;
            color: #7a1f1a;
            white-space: pre-wrap;
            font-family: 'Satoshi', Arial, sans-serif;
            font-size: 14px;
          }}

          .back {{
            position: absolute;
            right: 0;
            bottom: 0;
            font-size: 28px;
            color: #111;
            text-decoration: none;
          }}
          .back:hover {{
            text-decoration: underline;
          }}
        </style>
      </head>

      <body>
        <div class="wrap">
          <div class="left">
            <img class="heatmap" src="data:image/png;base64,{heatmap_png_b64}" />
          </div>

          <div class="right">
            <div class="cards">
              <div class="card">
                <div class="label">Input</div>
                <div class="content">{input_text}</div>
              </div>

              {err_html}

              <div class="card">
                <div class="label">SW Generated</div>
                <div class="content">{sw_text}</div>
              </div>

              <div class="card">
                <div class="label">HW Generated</div>
                <div class="content">{hw_text}</div>
              </div>
            </div>

            <a class="back" href="/attention_ui">Back</a>
          </div>
        </div>
      </body>
    </html>
    """
