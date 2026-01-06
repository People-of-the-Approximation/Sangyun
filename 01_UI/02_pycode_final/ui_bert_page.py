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
    err_blocks: str = "",  # Received but ignored in UI (not rendered)
):
    # ================= Logic Section =================
    # Determine if result is negative
    is_negative = hw_pneg >= hw_ppos

    # Set text label: NEGATIVE or POSITIVE
    big_label = "NEGATIVE" if is_negative else "POSITIVE"

    # Set color dynamically: NEG=Orange (#FC5230), POS=Green (#1B4F20)
    label_color = "#FC5230" if is_negative else "#1B4F20"

    # Format numbers as percentage strings (e.g. 0.952 -> "95.2%")
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
          /* CSS Variables for colors */
          :root {{
            --bg: #ffffff;      /* Background color */
            --text: #111;       /* Main text color */
            --muted: #6b7280;   /* Light grey text */
            --line: #e5e7eb;    /* Border color */
            --label: {label_color}; /* Dynamic Label Color (Orange/Green) */
          }}

          html, body {{
            height: 100%;
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: 'Satoshi', Arial, sans-serif;
          }}

          /* Main Container (Flexbox) */
          .wrap {{
            display: flex;
            height: 100vh;        /* Full viewport height */
            width: 100vw;         /* Full viewport width */
            padding: 24px 28px;   /* Padding: Top/Bottom 24px, Left/Right 28px */
            gap: 28px;            /* Gap between Left (Image) and Right (Text) sections */
            box-sizing: border-box;
          }}

          /* Left Section (Image Container) */
          .left {{
            display: flex;
            align-items: center;      /* Vertically center the image */
            justify-content: center;  /* Horizontally center the image */
          }}

          /* Heatmap Image Styling */
          .heatmap {{
            border: 1px solid var(--line);
            
            /* ✅ Max Height: Screen height minus padding (48px) */
            /* Decrease this if you want the image shorter vertically */
            max-height: calc(100vh - 48px);
            
            /* ✅ Max Width: 64% of screen width */
            /* Decrease "64vw" if you want the image narrower */
            max-width: 64vw;
            
            width: auto;
            height: auto;           /* Maintain aspect ratio */
            border-radius: 14px;    /* Rounded corners */
          }}

          /* Right Section (Text Container) */
          .right {{
            flex: 1 1 auto;       /* Fill remaining space */
            min-width: 360px;     /* Minimum width to prevent crushing text */
            position: relative;   /* Crucial for absolute positioning inside */
          }}

          /* Hide metadata (debug info) */
          .meta {{ display: none; }}

          /* ✅ Big Label Styling ("POSITIVE"/"NEGATIVE") */
          .big {{
            position: absolute;
            
            /* Vertical Position: 50% is center. 
               Change to 40% to move UP, 60% to move DOWN. */
            top: 50%;
            
            left: 50%;
            transform: translate(-50%, -50%); /* Perfect center adjustment */

            /* Font Size */
            font-size: 112px;
            
            font-weight: 500;
            letter-spacing: 1px;
            color: var(--label);
            line-height: 1.0;
            margin: 0;
            white-space: nowrap;  /* Prevent line wrapping */
          }}

          /* ✅ Stats Table Styling 
             Position logic: Center (50%) + Half Big Label (56px) + Gap (30px)
          */
          .stats {{
            position: absolute;
            
            /* "30px" determines distance below the Big Label. 
               Increase to 50px to push stats lower. */
            top: calc(50% + 112px / 2 + 30px);
            
            left: 50%;
            transform: translateX(-50%); /* Center horizontally */

            display: grid;
            
            /* Column Widths: Label (280px) | Value (Auto) */
            /* Adjust 280px if labels are wrapping */
            grid-template-columns: 280px 1fr;
            
            row-gap: 10px;    /* Vertical gap between rows */
            column-gap: 28px; /* Horizontal gap between columns */
          }}

          /* Key Label Style (e.g., "HW(approx)") */
          .k {{
            font-size: 25px;
            font-weight: 500;
            color: var(--muted);
            white-space: nowrap;
          }}

          /* Value Style (e.g., "95.2%") */
          .v {{
            font-size: 25px;
            font-weight: 500;
            color: var(--text);
            white-space: nowrap;
          }}

          /* "Back" Button Styling */
          .back {{
            position: absolute;
            right: 0;   /* Stick to right edge */
            bottom: 0;  /* Stick to bottom edge */
            
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

            <a class="back" href="/attention_ui">Back</a>
          </div>
        </div>
      </body>
    </html>
    """
