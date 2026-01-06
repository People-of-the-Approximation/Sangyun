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
    err_blocks: str = "",  # 받아도 UI에서는 아예 무시(렌더링 안 함)
):
    is_negative = hw_pneg >= hw_ppos
    big_label = "NEGATIVE" if is_negative else "POSITIVE"
    label_color = "#FC5230" if is_negative else "#1B4F20"  # NEG=주황, POS=초록

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
            border-radius: 14px;
          }}

          .right {{
            flex: 1 1 auto;
            min-width: 360px;
            position: relative;
          }}

          .meta {{ display: none; }}

          /* ✅ 오른쪽 영역에서 "라벨" 중앙 */
          .big {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);

            font-size: 112px;
            font-weight: 500;
            letter-spacing: 1px;
            color: var(--label);
            line-height: 1.0;
            margin: 0;
            white-space: nowrap;
          }}

          /*
            ✅ stats는 big보다 30px 아래에 고정
            - big의 기준이 top:50% 이고 transform(-50%)이므로
            - stats는 "center + 30px"로 명확히 분리
          */
          .stats {{
            position: absolute;
            top: calc(50% + 112px / 2 + 30px);
            left: 50%;
            transform: translateX(-50%);

            display: grid;
            grid-template-columns: 280px 1fr;
            row-gap: 10px;
            column-gap: 28px;
          }}

          .k {{
            font-size: 25px;
            font-weight: 500;
            color: var(--muted);
            white-space: nowrap;
          }}

          .v {{
            font-size: 25px;
            font-weight: 500;
            color: var(--text);
            white-space: nowrap;
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

            <!-- ✅ 에러 UI 완전 제거 (err_blocks 출력 안 함) -->

            <a class="back" href="/attention_ui">Back</a>
          </div>
        </div>
      </body>
    </html>
    """
