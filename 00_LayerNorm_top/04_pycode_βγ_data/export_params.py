import torch
import numpy as np
from transformers import BertModel


def float_to_hex_q10(val_float):
    """
    Float 값을 Q10 Fixed Point 16진수 문자열(4자리)로 변환
    예: 1.0 -> 1024 -> '0400', -1.0 -> -1024 -> 'fc00'
    """
    # 1. Scaling
    val_fixed = val_float * 1024.0

    # 2. Rounding & Clamping
    val_int = int(round(val_fixed))
    val_int = max(min(val_int, 32767), -32768)

    # 3. 2's Complement for Hex (Python trick: & 0xFFFF)
    hex_str = f"{val_int & 0xFFFF:04x}"
    return hex_str


# 1. 모델 로드
print("Loading BERT Model...")
model = BertModel.from_pretrained("bert-base-uncased")

# 2. 파라미터 추출 (768개)
# 편의상 첫번째 LayerNorm 가중치를 가져옵니다.
gamma = model.embeddings.LayerNorm.weight.detach().numpy()  # Shape: (768,)
beta = model.embeddings.LayerNorm.bias.detach().numpy()  # Shape: (768,)

print(f"Total Params: {len(gamma)}")
print("Generating wide-format .mem files (12 lines, 1024 bits each)...")


# 3. 파일 쓰기 함수 (Wide Format)
def write_wide_mem_file(filename, data):
    with open(filename, "w") as f:
        # 전체 768개를 64개씩 끊어서 12줄로 만듭니다.
        # Verilog 메모리가 [0:11] 깊이를 가지기 때문입니다.
        chunk_size = 64
        num_chunks = 12  # 768 / 64

        for i in range(num_chunks):
            # 1. 현재 라인에 들어갈 64개 데이터 슬라이싱
            chunk = data[i * chunk_size : (i + 1) * chunk_size]

            # 2. [중요] 역순 정렬 (Reverse)
            # Verilog 벡터 [1023:0]에 파일 텍스트를 넣을 때,
            # 오른쪽 끝 문자가 0번 비트(LSB, k=0)에 들어갑니다.
            # 따라서 k=63, k=62 ... k=0 순서로 문자열을 붙여야 합니다.
            line_str = ""
            for val in reversed(chunk):
                line_str += float_to_hex_q10(val)

            # 3. 파일에 쓰기
            f.write(f"{line_str}\n")


# 실행
write_wide_mem_file("bert_gamma.mem", gamma)
write_wide_mem_file("bert_beta.mem", beta)

print("Done! Check 'bert_gamma.mem' and 'bert_beta.mem'.")
