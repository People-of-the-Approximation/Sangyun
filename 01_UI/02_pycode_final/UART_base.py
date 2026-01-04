# UART_base.py
import serial
import time
import numpy as np

Q = 10
SCALE = 1 << Q
I16_MIN, I16_MAX = -32768, 32767
Q610_MIN, Q610_MAX = -32.0, (32.0 - 1.0 / SCALE)


def open_serial(port: str, baud: int = 115200, timeout: float = 1.0) -> serial.Serial:
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        write_timeout=timeout,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    # 포트 오픈 후 안정화 대기
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def send_exact(ser: serial.Serial, frame: bytes):
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    return


def read_exact(ser: serial.Serial, N: int, deadline_s: float = 2.0) -> bytes:
    end_time_s = time.perf_counter() + deadline_s
    buffer = bytearray()

    while len(buffer) < N:
        chunk = ser.read(N - len(buffer))
        if chunk:
            buffer.extend(chunk)
        else:
            if time.perf_counter() > end_time_s:
                raise TimeoutError(
                    f"Error: timeout -> read_exact() [{len(buffer)}/{N}]"
                )
    return bytes(buffer)


def floats_to_q610_bytes(x64, *, endian: str = "big", mode: str = "saturate") -> bytes:
    # [수정] FPGA에 맞춰 Big Endian 기본값 설정
    x = np.asarray(x64, dtype=np.float64)
    if x.shape != (64,):
        raise ValueError("Input must be a 1D array of length 64.")

    if mode == "strict":
        if np.any(x < Q610_MIN) or np.any(x > Q610_MAX):
            idx = int(np.where((x < Q610_MIN) | (x > Q610_MAX))[0][0])
            raise OverflowError(
                f"Q6.10 range exceeded (index={idx}, value={x[idx]:.6f})"
            )

    scaled = np.rint(x * SCALE).astype(np.int32)
    scaled = np.clip(scaled, I16_MIN, I16_MAX).astype(np.int16)

    dtype = ">i2" if endian == "big" else "<i2"
    return scaled.astype(dtype, copy=False).tobytes()


def q610_bytes_to_floats(b: bytes, *, endian: str = "big") -> np.ndarray:
    # [수정] 129바이트(헤더 1 + 데이터 128) 처리 로직 추가
    if len(b) == 129:
        actual_data = b[1:]  # 헤더/모드 바이트 제거
    elif len(b) == 128:
        actual_data = b
    else:
        # 데이터가 연속으로 들어올 때(Batch) 129의 배수가 아닐 수도 있으므로 체크
        if len(b) % 129 == 0:
            # 여기서는 단일 프레임 변환용이므로 일단 에러 처리하거나,
            # 외부에서 잘라서 호출해야 함을 가정
            pass
        # 일반적인 128바이트 변환 시도
        actual_data = b

    # 128바이트여야 64개 float 변환 가능
    if len(actual_data) != 128:
        raise ValueError(
            f"Invalid data length for float conversion: {len(actual_data)}"
        )

    dtype = ">i2" if endian == "big" else "<i2"
    i16 = np.frombuffer(actual_data, dtype=dtype)
    return i16.astype(np.float64) / SCALE


def build_softmax_frame(
    payload64: np.ndarray, header_val: int, *, endian: str = "big"
) -> bytes:
    # [수정] header_val을 첫 바이트로 붙여서 129바이트 프레임 생성
    payload_bytes = floats_to_q610_bytes(payload64, endian=endian)
    return bytes([header_val & 0xFF]) + payload_bytes
