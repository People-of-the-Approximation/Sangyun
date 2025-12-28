`timescale 1ns / 1ps

module tb_ln_stage3_normalize;

    // =============================================================
    // 1. 신호 선언
    // =============================================================
    reg          i_clk;
    reg          i_en;
    reg          i_valid_trigger;
    
    // Stage 2에서 넘어온 통계치
    reg signed [31:0] i_mean;
    reg signed [16:0] i_inv_sqrt;
    
    // 벡터 데이터
    reg [1023:0] i_raw_data_flat;
    reg [1023:0] i_gamma_flat;
    reg [1023:0] i_beta_flat;

    // 출력
    wire [1023:0] o_res_data_flat;
    wire          o_res_valid;

    // =============================================================
    // 2. DUT 연결
    // =============================================================
    ln_stage3_normalize dut (
        .i_clk(i_clk),
        .i_en(i_en),
        .i_valid_trigger(i_valid_trigger),
        .i_mean(i_mean),
        .i_inv_sqrt(i_inv_sqrt),
        .i_raw_data_flat(i_raw_data_flat),
        .i_gamma_flat(i_gamma_flat),
        .i_beta_flat(i_beta_flat),
        .o_res_data_flat(o_res_data_flat),
        .o_res_valid(o_res_valid)
    );

    // =============================================================
    // 3. 클럭 생성
    // =============================================================
    initial begin
        i_clk = 0;
        forever #5 i_clk = ~i_clk;
    end

    // =============================================================
    // 4. 스트리밍 테스트 시나리오
    // =============================================================
    integer k, ch; // 루프 변수

    initial begin
        // 초기화
        i_en = 0;
        i_valid_trigger = 0;
        i_mean = 0; i_inv_sqrt = 0;
        i_raw_data_flat = 0; i_gamma_flat = 0; i_beta_flat = 0;

        #100;
        @(posedge i_clk);
        i_en = 1;

        $display("=== Stage 3 Streaming Test Start (Throughput = 1) ===");
        $display("feeding 10 packets consecutively...");

        // ---------------------------------------------------------
        // [루프] 10번 연속으로 데이터 밀어넣기
        // ---------------------------------------------------------
        for (k = 1; k <= 10; k = k + 1) begin
            @(posedge i_clk);
            i_valid_trigger <= 1;

            // 1. 통계치 변화 (데이터 구분을 위해)
            // Mean을 k * 10 으로 설정 (10, 20, 30...)
            i_mean     <= k * 10;       
            i_inv_sqrt <= 17'd1024; // 1.0 고정

            // 2. 64개 채널 데이터 채우기
            for (ch = 0; ch < 64; ch = ch + 1) begin
                // 입력 데이터: 100 + k + ch (채널마다 조금씩 다르게)
                // 예: k=1이면 101, 102...
                i_raw_data_flat[16*ch +: 16] = 100 + k + ch;
                
                // Gamma=1, Beta=0
                i_gamma_flat[16*ch +: 16]    = 16'd1024; 
                i_beta_flat[16*ch +: 16]     = 16'd0;    
            end
        end

        // ---------------------------------------------------------
        // 입력 종료
        // ---------------------------------------------------------
        @(posedge i_clk);
        i_valid_trigger <= 0;
        i_raw_data_flat <= 0; // 보기 좋게 초기화

        // 결과 대기
        #100;
        $finish;
    end

    // =============================================================
    // 5. 결과 모니터링
    // =============================================================
    integer out_cnt = 0;
    reg signed [15:0] check_val_ch0;

    always @(posedge i_clk) begin
        if (o_res_valid) begin
            out_cnt = out_cnt + 1;
            
            // 첫 번째 채널(Channel 0)의 값만 뽑아서 확인
            check_val_ch0 = o_res_data_flat[15:0];

            $display("[Time %0t] Output #%2d Valid! | Ch0 Value: %d", 
                     $time, out_cnt, check_val_ch0);
        end
    end

endmodule