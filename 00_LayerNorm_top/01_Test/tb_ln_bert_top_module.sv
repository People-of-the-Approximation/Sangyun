`timescale 1ns / 1ps

module tb_ln_bert_top_module;

    // =============================================================
    // 1. 신호 선언
    // =============================================================
    reg          i_clk;
    reg          i_rst;
    reg          i_en;
    
    reg          i_valid;
    reg [1023:0] i_data_flat; // 1024비트 입력

    wire [1023:0] o_result_flat;
    wire          o_valid;

    // =============================================================
    // 2. DUT (Device Under Test) 연결
    // =============================================================
    ln_bert_top_module dut (
        .i_clk(i_clk),
        .i_rst(i_rst),
        .i_en(i_en),
        .i_valid(i_valid),
        .i_data_flat(i_data_flat),
        .o_result_flat(o_result_flat),
        .o_valid(o_valid)
    );

    // =============================================================
    // 3. 클럭 생성 (100MHz)
    // =============================================================
    initial begin
        i_clk = 0;
        forever #5 i_clk = ~i_clk;
    end

    // =============================================================
    // 4. 테스트 시나리오
    // =============================================================
    integer k, ch;

    initial begin
        // ---------------------------------------------------------
        // [Step 1] 초기화
        // ---------------------------------------------------------
        i_rst = 1;
        i_en = 0;
        i_valid = 0;
        i_data_flat = 0;

        #100; // Reset 유지
        i_rst = 0;
        #20;
        i_en = 1; // 모듈 Enable

        $display("=== Top Module Test Start ===");
        $display("Feeding 120 packets (10 Banks) to verify full throughput...");

        // ---------------------------------------------------------
        // [Step 2] 데이터 입력 (120개 = 10 Banks)
        // Bank Size(12) * 10 = 120개의 데이터를 연속으로 밀어넣습니다.
        // 병목이 해결되었다면, 입력이 끝날 때까지 멈춤 없이 들어가야 합니다.
        // ---------------------------------------------------------
        for (k = 0; k < 120; k = k + 1) begin
            
            // 데이터 생성: 각 채널에 (k값 + 채널인덱스)를 넣음
            // 예: 
            // Packet 0: 1000, 1001, ...
            // Packet 1: 1010, 1011, ...
            // Packet 2: 1020, 1021, ...
            // 데이터 값이 선형적으로 증가하므로, LayerNorm 결과는 일정하게 수렴해야 함
            for (ch = 0; ch < 64; ch = ch + 1) begin
                i_data_flat[16*ch +: 16] = 16'd1000 + (k * 10) + ch; 
            end

            i_valid = 1;
            @(posedge i_clk); // 1클럭마다 연속 입력 (Throughput Test)
        end

        // ---------------------------------------------------------
        // [Step 3] 입력 종료 및 대기
        // ---------------------------------------------------------
        i_valid = 0;
        i_data_flat = 0;
        
        $display("Input Finished (120 Packets). Waiting for results...");

        // 파이프라인(4 Stages)을 모두 통과하여 결과가 다 나올 때까지 충분히 대기
        // 예상 Latency: 12(S1) + 12(S2) + 12(S3) + 12(S4) + @ = 약 50~60클럭 후 시작
        // 출력 120개가 다 나오려면 120클럭 이상 더 필요함
        #3000; 
            // 시뮬레이션 끝날 때 최종 성적표 출력
            // initial 블록의 $finish 직전에 추가
    
        if (err_cnt == 0) begin
            $display("\n\n========================================");
            $display("   ALL PASSED! (120/120 Packets Matched)");
            $display("========================================");
        end else begin
            $display("\n\n========================================");
            $display("   FAIL: Found %0d mismatches.", err_cnt);
            $display("========================================");
        end
        $finish;
        
        $display("Simulation Finished.");
        $finish;
    end

    // =============================================================
    // 5. 결과 모니터링
    // =============================================================
reg [1023:0] golden_mem [0:119]; // 파이썬 결과 담을 메모리 (120줄)
    integer out_cnt = 0;
    integer err_cnt = 0;

    initial begin
        // 파이썬이 만든 헥사 파일 읽기 (같은 폴더에 두세요)
        $readmemh("golden_output_hex.txt", golden_mem);
    end

    always @(posedge i_clk) begin
        if (o_valid) begin
            // 1. 결과 비교
            if (o_result_flat !== golden_mem[out_cnt]) begin
                $display("[ERROR] Mismatch at Packet #%0d", out_cnt);
                $display("  Expected (Py): %h", golden_mem[out_cnt]);
                $display("  Actual   (HW): %h", o_result_flat);
                err_cnt = err_cnt + 1;
            end else begin
                // 잘 맞으면 점(.) 하나 찍어서 진행 상황 표시
                $write("."); 
            end

            out_cnt = out_cnt + 1;
        end
    end    

endmodule