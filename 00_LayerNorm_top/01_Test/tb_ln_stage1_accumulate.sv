`timescale 1ns / 1ps

module tb_ln_stage1_accumulate;

    // =============================================================
    // 1. 신호 선언
    // =============================================================
    reg          i_clk;
    reg          i_rst;
    reg          i_en;
    reg          i_valid;
    reg [1023:0] i_data_flat;
    
    reg [1:0]    i_ptr_in;
    reg [3:0]    i_cycle_cnt;

    wire         o_acc_valid;
    wire [1:0]   o_acc_ptr;
    wire [3:0]   o_acc_cnt;
    wire signed [21:0] o_part_sum;
    wire signed [37:0] o_part_sq_sum;

    // =============================================================
    // 2. 클럭 생성 (100MHz)
    // =============================================================
    initial begin
        i_clk = 0;
        forever #5 i_clk = ~i_clk;
    end

    // =============================================================
    // 3. DUT (Device Under Test) 인스턴스
    // =============================================================
    ln_stage1_accumulate dut (
        .i_clk(i_clk),
        .i_rst(i_rst),
        .i_en(i_en),
        .i_valid(i_valid),
        .i_data_flat(i_data_flat),
        .i_ptr_in(i_ptr_in),
        .i_cycle_cnt(i_cycle_cnt),
        .o_acc_valid(o_acc_valid),
        .o_acc_ptr(o_acc_ptr),
        .o_acc_cnt(o_acc_cnt),
        .o_part_sum(o_part_sum),
        .o_part_sq_sum(o_part_sq_sum)
    );

    // =============================================================
    // 4. 테스트 시나리오
    // =============================================================
    integer k;

    initial begin
        // 1. 초기화
        i_rst = 1;
        i_en = 0;
        i_valid = 0;
        i_data_flat = 0;
        i_ptr_in = 0;
        i_cycle_cnt = 0;

        // 2. 리셋 해제
        #100;
        @(posedge i_clk);
        i_rst = 0;
        i_en = 1; // 모듈 활성화
        
        $display("=== Simulation Start ===");

        // 3. 데이터 입력 시나리오 (12 Cycle 연속 입력)
        // 가정: cycle_cnt는 0~11, 입력 데이터는 모든 채널에 동일한 값 'k'를 입력
        // 예: k=1이면 64개 채널 모두 1 -> Sum=64, SqSum=64
        // 예: k=2이면 64개 채널 모두 2 -> Sum=128, SqSum=256 (4*64)
        
        $display("[Time %0t] Input Sequence Start", $time);
        
        for (k = 0; k < 12; k = k + 1) begin
            @(posedge i_clk);
            i_valid <= 1;
            i_ptr_in <= 2'd0;       // Bank 0
            i_cycle_cnt <= k[3:0];  // 0 ~ 11
            
            // 모든 64개 입력에 k 값을 채움
            i_data_flat <= {(1024/16){k[15:0]}}; 
        end

        // 4. 입력 중단 (Idle)
        @(posedge i_clk);
        i_valid <= 0;
        i_data_flat <= 0;

        // 5. 대기 (Latency 7클럭 + 여유분 확인)
        repeat(15) @(posedge i_clk);

        $display("=== Simulation End ===");
        $finish;
    end

    // =============================================================
    // 5. 결과 검증 (Scoreboard)
    // =============================================================
    // 예상되는 결과값을 계산하기 위한 변수
    reg signed [21:0] expected_sum;
    reg signed [37:0] expected_sq_sum;

    always @(posedge i_clk) begin
        if (o_acc_valid) begin
            // 입력 데이터가 모든 채널에 대해 'cnt' 값이었으므로,
            // Sum = cnt * 64
            // SqSum = (cnt * cnt) * 64
            
            expected_sum    = o_acc_cnt * 64;
            expected_sq_sum = (o_acc_cnt * o_acc_cnt) * 64;

            $display("[Time %0t] Output Valid! Cnt: %2d", $time, o_acc_cnt);
            $display("    -> Sum   : Output=%6d / Expected=%6d", o_part_sum, expected_sum);
            $display("    -> SqSum : Output=%6d / Expected=%6d", o_part_sq_sum, expected_sq_sum);
            
            if ((o_part_sum === expected_sum) && (o_part_sq_sum === expected_sq_sum)) begin
                 $display("    -> [PASS] Result matches.");
            end else begin
                 $display("    -> [FAIL] Mismatch detected!");
                 $stop; // 에러 발생 시 시뮬레이션 중단
            end
        end
    end

endmodule