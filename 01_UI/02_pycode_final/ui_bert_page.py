# ui_bert_page.py


def render_bert_result_page(
    *,
    used_mode: str,
    layer: int,
    head: int,
    T: int,
    attn_id: str,
    hw_ppos: float,
    hw_pneg: float,
    sw_ppos: float,
    sw_pneg: float,
    match_line: str,
    err_blocks: str = "",
):
    is_negative = hw_pneg >= hw_ppos
    big_label = "NEGATIVE" if is_negative else "POSITIVE"
    label_color = "#FC5230" if is_negative else "#1B4F20"  # ✅ POS=초록, NEG=주황

    hw_ppos_s = f"{hw_ppos*100:.1f}%"
    hw_pneg_s = f"{hw_pneg*100:.1f}%"
    sw_ppos_s = f"{sw_ppos*100:.1f}%"
    sw_pneg_s = f"{sw_pneg*100:.1f}%"

    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>BERT Attention</title>

        <!-- ✅ Satoshi 로드 (static/style.css에 @font-face 있음) -->
        <link rel="stylesheet" href="/static/style.css" />

        <style>
          :root {{
            --bg: #ffffff;
            --text: #111;
            --muted: #6b7280;
            --line: #e5e7eb;
            --label: {label_color};
          }}

          html, body {{
            height: 100%;
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: 'Satoshi', Arial, sans-serif; /* ✅ 전부 Satoshi */
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
          }}

          .right {{
            flex: 1 1 auto;
            min-width: 360px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            font-family: 'Satoshi', Arial, sans-serif; /* ✅ 명시 */
          }}

          /* ✅ 상단 meta 텍스트 완전히 비활성화 */
          .meta {{
            display: none;
          }}

          /* ✅ POSITIVE/NEGATIVE 위치 고정 (에러 유무로 흔들리지 않게) */
          .big {{
            position: absolute;
            top: 90px;      /* 필요하면 여기만 조절 */
            left: 0;

            font-family: 'Satoshi', Arial, sans-serif;
            font-size: 92px;
            font-weight: 500;     /* Satoshi Medium 톤 */
            letter-spacing: 1px;
            color: var(--label);
            line-height: 1.0;
            margin: 0;
          }}

          /* ✅ 텍스트 영역: Satoshi + 크기 조절 */
          .stats {{
            display: grid;
            grid-template-columns: 200px 1fr;
            row-gap: 10px;
            column-gap: 20px;

            margin-top: 220px;    /* big 아래로 내려줌 */
            font-family: 'Satoshi', Arial, sans-serif;
          }}

          .k {{
            font-family: 'Satoshi', Arial, sans-serif;
            font-size: 18px;
            font-weight: 500;
            color: var(--muted);
          }}

          .v {{
            font-family: 'Satoshi', Arial, sans-serif;
            font-size: 22px;
            font-weight: 500;
            color: var(--text);
          }}

          .err {{
            margin-top: 18px;
            font-family: 'Satoshi', Arial, sans-serif;
          }}

          .back {{
            position: absolute;
            right: 0;
            bottom: 0;
            font-size: 28px;
            color: #111;
            text-decoration: none;
            font-family: 'Satoshi', Arial, sans-serif;
          }}

          .back:hover {{
            text-decoration: underline;
          }}
        </style>
      </head>

      <body>
        <div class="wrap">
          <div class="left">
            <img class="heatmap" src="/attn_heatmap.png?id={attn_id}" />
          </div>

          <div class="right">
            <div class="meta">Attention heatmap (mode={used_mode}, layer={int(layer)}, head={int(head)}, T={int(T)})</div>

            <div class="big">{big_label}</div>

            <div class="stats">
              <div class="k">HW(approx)</div>
              <div class="v">Ppos : {hw_ppos_s}, Pneg : {hw_pneg_s}</div>

              <div class="k">SW(baseline)</div>
              <div class="v">Ppos : {sw_ppos_s}, Pneg : {sw_pneg_s}</div>

              <div class="k">Prediction Difference</div>
              <div class="v">{match_line}</div>
            </div>

            <div class="err">{err_blocks}</div>

            <a class="back" href="/attention_ui">Back</a>
          </div>
        </div>
      </body>
    </html>
    """
