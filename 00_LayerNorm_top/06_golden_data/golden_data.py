import numpy as np
import os

# ==============================================================================
# 설정
# ==============================================================================
NUM_PACKETS = 120
CHANNELS = 64
INPUT_START = 1000
INPUT_STEP = 10
FRAC_BITS = 10  # Q10 포맷 (S5.10)
BANK_SIZE = 12  # [핵심] 하드웨어는 12개를 한 묶음으로 처리함


# ==============================================================================
# Helper Functions
# ==============================================================================
def to_fixed_hex(float_val, frac_bits):
    scaled = float_val * (2**frac_bits)
    scaled_int = int(round(scaled))
    if scaled_int > 32767:
        scaled_int = 32767
    if scaled_int < -32768:
        scaled_int = -32768
    return f"{(scaled_int & 0xFFFF):04x}"


def load_mem_file_fixed(filename, num_lines=12):
    # (파일 로딩 로직은 기존과 동일하되, Q10 값을 Float로 변환하여 리턴)
    # 파일이 없으면 1.0 리턴
    if not os.path.exists(filename):
        return np.ones((num_lines, CHANNELS))
    # ... (생략: 파일 읽어서 / 1024.0 해서 리턴) ...
    # 편의상 여기선 1.0(Gamma), 0.0(Beta) 쓴다고 가정
    return np.ones((num_lines, CHANNELS)), np.zeros((num_lines, CHANNELS))


# ==============================================================================
# Main Simulation (Block Mode)
# ==============================================================================
def run_hardware_match():
    # 1. 감마/베타 로드 (없으면 기본값 1.0, 0.0)
    # 실제로는 파일에서 읽어와야 함
    gamma_bank = np.ones((12, 64))
    beta_bank = np.zeros((12, 64))

    print(f"Generating Golden Data (Hardware Match: Block Size {BANK_SIZE})...")

    # 120개 패킷 생성
    all_packets = []
    for k in range(NUM_PACKETS):
        pkt = np.array(
            [INPUT_START + (k * INPUT_STEP) + ch for ch in range(CHANNELS)],
            dtype=np.float64,
        )
        all_packets.append(pkt)

    # 12개씩 묶어서 처리 (Hardware Logic)
    for bank_idx in range(0, NUM_PACKETS, BANK_SIZE):
        # 1. 뱅크(12개 패킷) 데이터 수집
        bank_data = np.array(
            all_packets[bank_idx : bank_idx + BANK_SIZE]
        )  # shape: (12, 64)

        # 2. [핵심] 12개 전체에 대한 평균과 분산 계산 (Stage 1 & 2 Logic)
        # 하드웨어는 12*64 = 768개 전체의 합을 구함
        block_mean = np.mean(bank_data)
        block_var = np.var(bank_data)
        block_inv_sqrt = 1.0 / np.sqrt(block_var + 1e-5)

        print(
            f"[Bank {bank_idx//12}] Global Mean: {block_mean:.2f}, InvSqrt: {block_inv_sqrt:.6f}"
        )

        # 3. 각 패킷별로 정규화 수행 (Stage 4 Logic)
        for i in range(BANK_SIZE):
            cycle = i  # 0~11
            curr_pkt = bank_data[i]
            curr_gamma = gamma_bank[cycle]
            curr_beta = beta_bank[cycle]

            # (x - GlobalMean) * GlobalInvSqrt * Gamma + Beta
            norm_pkt = (curr_pkt - block_mean) * block_inv_sqrt * curr_gamma + curr_beta

            # Ch0 값 출력 (비교용)
            packet_global_idx = bank_idx + i
            hex_val = to_fixed_hex(norm_pkt[0], FRAC_BITS)

            # Verilog 결과와 비교할 수 있게 로그 출력
            if packet_global_idx >= 140:  # 안정화된 구간 확인
                print(
                    f"Packet {packet_global_idx}: Ch0 Float={norm_pkt[0]:.4f} -> Hex={hex_val}"
                )


if __name__ == "__main__":
    run_hardware_match()
