# Attention_approx.py
import numpy as np
import serial
import time
from UART_base import build_softmax_frame, send_exact, read_exact, q610_bytes_to_floats

# FPGA 메모리 한계 (안전하게 64로 설정)
FPGA_MAX_FRAME_DEPTH = 64


def _pack_params(L: int):
    # 길이에 따른 패킹 파라미터 결정
    if not (1 <= L <= 64):
        raise ValueError("Length must be between 1 and 64.")
    if L <= 16:
        return 16, 4  # 16개씩 4묶음 (Mode 0)
    elif L <= 32:
        return 32, 2  # 32개씩 2묶음 (Mode 1)
    else:
        return 64, 1  # 64개씩 1묶음 (Mode 2)


def softmax_FPGA_UART_batch(
    ser: serial.Serial,
    scores_list,
    *,
    pad_value: float = -32.0,
    deadline_s: float = 5.0,
):
    seqs = [np.asarray(s, dtype=np.float64) for s in scores_list]
    if not seqs:
        return []

    L = int(seqs[0].shape[0])
    if any(int(s.shape[0]) != L for s in seqs):
        raise ValueError("All sequences must have the same length.")

    # 1. 패킹 전략 수립
    block_size, max_pack = _pack_params(L)

    # FPGA Mode 값 결정
    if L <= 16:
        mode_val = 0
    elif L <= 32:
        mode_val = 1
    else:
        mode_val = 2

    # 2. 데이터 패킹 (Packing)
    payloads = []
    meta_info = []

    for off in range(0, len(seqs), max_pack):
        chunk = seqs[off : off + max_pack]
        G = len(chunk)

        payload = np.full(64, pad_value, dtype=np.float64)
        for g, vec in enumerate(chunk):
            start = g * block_size
            payload[start : start + L] = vec

        payloads.append(payload)
        meta_info.append(G)

    # 3. 배치 전송 및 수신
    final_results = []

    # FPGA 메모리 한계만큼 끊어서 처리
    for i in range(0, len(payloads), FPGA_MAX_FRAME_DEPTH):
        batch_payloads = payloads[i : i + FPGA_MAX_FRAME_DEPTH]
        batch_meta = meta_info[i : i + FPGA_MAX_FRAME_DEPTH]

        num_frames_to_send = len(batch_payloads)

        # [핵심] Depth Byte 전송: (보낼 프레임 수 - 1)
        depth_byte = num_frames_to_send - 1
        ser.write(bytes([depth_byte]))
        time.sleep(0.02)  # 상태 전이 대기

        # 프레임 연속 전송
        for payload in batch_payloads:
            frame = build_softmax_frame(payload, header_val=mode_val, endian="big")
            ser.write(frame)
            time.sleep(0.002)  # 안정성 확보

        # 결과 수신 (프레임 수 * 129바이트)
        expected_bytes = num_frames_to_send * 129
        rx_data = read_exact(ser, expected_bytes, deadline_s=deadline_s)

        # 결과 언패킹
        for row_idx in range(num_frames_to_send):
            # 129바이트 단위로 자르기
            chunk_bytes = rx_data[row_idx * 129 : (row_idx + 1) * 129]
            # 실수 변환
            probs64 = q610_bytes_to_floats(chunk_bytes, endian="big")

            # 원래 시퀀스로 분리
            num_seqs_in_row = batch_meta[row_idx]
            for g in range(num_seqs_in_row):
                start = g * block_size
                final_results.append(
                    np.asarray(probs64[start : start + L], dtype=np.float64)
                )

    return final_results


def attention(
    Q,
    K,
    V,
    ser: serial.Serial,
    *,
    pad_value: float = -32.0,
    deadline_s: float = 5.0,
    return_attn: bool = False,
):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    Nq, d_kq = Q.shape
    Nk, d_kk = K.shape
    Nv, d_kv = V.shape

    # 차원 검사
    assert Nq == Nk and d_kq == d_kk, "Dim Error: Q,K mismatch"
    assert Nv == Nk, "Dim Error: V rows must match K rows"
    # 하드웨어 제약사항 (64 이하만 처리)
    if not (1 <= Nk <= 64):
        # 64보다 크면 SW fallback 혹은 에러 처리해야 함. 여기서는 에러 발생.
        raise ValueError(f"Length N must be between 1 and 64 (got {Nk}).")

    d_k = d_kq

    # 1. Score 계산 (CPU)
    S_matrix = (Q @ K.T) / np.sqrt(d_k)

    # 2. 리스트 변환
    seqs = [S_matrix[i, :] for i in range(Nq)]

    # 3. FPGA 가속 (배치 처리)
    probs_list = softmax_FPGA_UART_batch(
        ser, seqs, pad_value=pad_value, deadline_s=deadline_s
    )

    # 4. 결과 합치기
    F = np.vstack(probs_list)

    # 5. Output 계산 (CPU)
    outputs = F @ V

    # 웹 UI 시각화를 위해 Attention Matrix(확률값)도 반환 가능하도록 처리
    if return_attn:
        return outputs, F

    return outputs
