import numpy as np
import os

# ==============================================================================
# 1. 시뮬레이션 설정 (Configuration)
# ==============================================================================
NUM_PACKETS = 120  # 총 패킷 수 (10 Banks * 12 Packets)
CHANNELS = 64  # 채널 수 (Feature Dimension)
BANK_SIZE = 12  # 하드웨어 뱅크 크기 (평균/분산 공유 단위)
FRAC_BITS = 10  # Q10 포맷 (S5.10)

# 입력 데이터 생성 패턴 (Verilog Testbench와 동일해야 함)
INPUT_START = 1000
INPUT_STEP = 10

# 파일 이름 설정
GAMMA_FILE = "bert_gamma.mem"
BETA_FILE = "bert_beta.mem"
OUTPUT_FILE = "golden_output_hex.txt"


# ==============================================================================
# 2. 헬퍼 함수: Q10 변환 및 파일 로딩
# ==============================================================================
def to_fixed_hex(float_val, frac_bits):
    """
    실수(Float)를 16비트 고정소수점 Hex(4글자)로 변환
    - Range: -32768 ~ 32767
    - Format: 2's Complement Hex
    """
    # 1. Scaling
    scaled = float_val * (2**frac_bits)

    # 2. Rounding
    scaled_int = int(round(scaled))

    # 3. Saturation (Clamping)
    if scaled_int > 32767:
        scaled_int = 32767
    elif scaled_int < -32768:
        scaled_int = -32768

    # 4. Convert to 4-digit Hex (handle negative with Mask)
    return f"{(scaled_int & 0xFFFF):04x}"


def load_mem_file(filename, rows=12, cols=64, frac_bits=10):
    """
    Verilog .mem 파일을 읽어서 (12, 64) 형태의 Float Array로 변환
    - 파일이 없으면 Gamma=1.0, Beta=0.0 기본값 반환
    - 파일이 있으면 16비트 Hex 스트링을 파싱하여 Float로 변환
    """
    # 기본값 결정
    is_gamma = "gamma" in filename
    default_val = 1.0 if is_gamma else 0.0

    if not os.path.exists(filename):
        print(f"[Info] '{filename}' not found. Using default value {default_val}.")
        return np.full((rows, cols), default_val)

    try:
        with open(filename, "r") as f:
            # 공백, 개행 제거 후 한 줄로 합침
            content = f.read().split()
            hex_str = "".join(content)

        vals = []
        # 4글자(16비트)씩 끊어서 읽기
        for i in range(0, len(hex_str), 4):
            chunk = hex_str[i : i + 4]
            if len(chunk) < 4:
                break

            # Hex -> Signed Int
            val_int = int(chunk, 16)
            if val_int > 32767:
                val_int -= 65536

            # Int -> Float
            vals.append(val_int / (2**frac_bits))

        # 데이터 개수가 부족하면 채우기
        required = rows * cols
        if len(vals) < required:
            vals.extend([default_val] * (required - len(vals)))

        # 하드웨어 mem 파일은 보통 MSB(Ch63)부터 쓰여있을 확률이 높음.
        # 하지만 Python 계산 편의를 위해 일단 (12, 64)로 Reshape
        # *주의* 만약 결과가 이상하면 여기서 np.flip(arr, axis=1)을 고려해야 함.
        # 현재는 일반적인 순서(Ch0...Ch63)로 가정하고 로드함.
        return np.array(vals[:required]).reshape(rows, cols)

    except Exception as e:
        print(f"[Error] Failed to parse {filename}: {e}")
        return np.full((rows, cols), default_val)


# ==============================================================================
# 3. 메인 시뮬레이션 (Layer Norm Calculation)
# ==============================================================================
def run_simulation():
    print(f"=== Generating Golden Data ({OUTPUT_FILE}) ===")

    # 1. 가중치 파일 로드
    gamma_table = load_mem_file(GAMMA_FILE)
    beta_table = load_mem_file(BETA_FILE)

    # 2. 입력 데이터 생성 (Verilog Testbench와 동일 로직)
    # Packet k의 Ch c 값 = 1000 + 10*k + c
    all_packets = []
    for k in range(NUM_PACKETS):
        pkt = np.array(
            [INPUT_START + (k * INPUT_STEP) + ch for ch in range(CHANNELS)],
            dtype=np.float64,
        )
        all_packets.append(pkt)
    all_packets = np.array(all_packets)  # (120, 64)

    golden_lines = []

    # 3. Bank 단위(12개씩) 처리
    for bank_idx in range(0, NUM_PACKETS, BANK_SIZE):
        # (1) Bank 데이터 슬라이싱
        bank_data = all_packets[bank_idx : bank_idx + BANK_SIZE]

        # (2) Global Mean/Var 계산 (하드웨어 로직: 12*64개 전체 통합)
        block_mean = np.mean(bank_data)
        block_var = np.var(bank_data)
        block_inv_sqrt = 1.0 / np.sqrt(block_var + 1e-5)  # Epsilon

        # 디버깅 출력 (첫 번째 뱅크만)
        if bank_idx == 0:
            print(f"[Bank 0] Mean: {block_mean:.2f}, InvSqrt: {block_inv_sqrt:.5f}")
            print(f"[Bank 0] Gamma[0][0]: {gamma_table[0][0]}")

        # (3) 정규화 및 Hex String 생성
        for i in range(BANK_SIZE):
            cycle = i  # 0~11

            # 원본 데이터 및 가중치
            raw_pkt = bank_data[i]
            curr_gamma = gamma_table[cycle]
            curr_beta = beta_table[cycle]

            # Layer Norm 수식
            norm_pkt = (raw_pkt - block_mean) * block_inv_sqrt * curr_gamma + curr_beta

            # (4) [핵심 수정] Output Hex String 생성 (MSB First)
            # 하드웨어 Bus [1023:0]은 Channel 63이 가장 왼쪽(MSB)에 위치함.
            # 따라서 63, 62, ... 1, 0 순서로 Hex를 생성해서 붙여야 함.
            hex_parts = []
            for ch in reversed(range(CHANNELS)):  # <--- 여기가 핵심! 역순!
                hex_val = to_fixed_hex(norm_pkt[ch], FRAC_BITS)
                hex_parts.append(hex_val)

            # 64개 채널을 이어 붙여 1개의 긴 문자열(Line) 생성
            line_str = "".join(hex_parts)
            golden_lines.append(line_str)

    # 4. 파일 저장
    with open(OUTPUT_FILE, "w") as f:
        for line in golden_lines:
            f.write(line + "\n")

    print(f"Done! Generated {len(golden_lines)} lines in '{OUTPUT_FILE}'.")
    print("Please copy this file to your Vivado simulation directory.")


if __name__ == "__main__":
    run_simulation()
