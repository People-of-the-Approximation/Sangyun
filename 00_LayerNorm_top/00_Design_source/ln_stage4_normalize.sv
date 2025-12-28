`timescale 1ns / 1ps

module ln_stage4_normalize (
    input  wire i_clk,
    input  wire i_en,
    input  wire i_valid_trigger, 
    
    // [추가됨] 입력 데이터의 주소 (Tag/Cycle Count) -> 6비트 (Bank 2bit + Cycle 4bit)
    input  wire [5:0] i_addr, 
    
    input  wire signed [31:0] i_mean,
    input  wire signed [16:0] i_inv_sqrt, 
    
    input  wire [1023:0] i_raw_data_flat,
    input  wire [1023:0] i_gamma_flat, 
    input  wire [1023:0] i_beta_flat,  

    output reg  [1023:0] o_res_data_flat,
    output reg           o_res_valid,

    // [추가됨] 파이프라인을 통과해 나온 주소 -> 6비트
    output reg  [5:0] o_res_addr
);

    // =============================================================
    // 1. Data Unpacking (Flat -> Array)
    // =============================================================
    wire signed [15:0] w_raw   [0:63];
    wire signed [15:0] w_gamma [0:63];
    wire signed [15:0] w_beta  [0:63];

    genvar g;
    generate
        for (g=0; g<64; g=g+1) begin : unpack_loop
            assign w_raw[g]   = i_raw_data_flat[16*g +: 16];
            assign w_gamma[g] = i_gamma_flat[16*g +: 16];
            assign w_beta[g]  = i_beta_flat[16*g +: 16];
        end
    endgenerate

    // =============================================================
    // 2. Control Signal & Address Delay (Valid + Address 파이프라인)
    // =============================================================
    // IP Latency 설정
    localparam LAT_MULT1 = 5; 
    localparam LAT_MULT2 = 5; 
    
    // Total Pipeline Depth = 1(Input Reg) + 5(Mult1) + 5(Mult2) = 11
    localparam TOTAL_PIPE_DELAY = 1 + LAT_MULT1 + LAT_MULT2; 

    reg [TOTAL_PIPE_DELAY-1:0] r_valid_pipe;
    
    // [수정완료] 주소(Address)를 전달할 파이프라인 (폭: 6bit로 수정됨!)
    reg [5:0] r_addr_pipe [0:TOTAL_PIPE_DELAY-1];
    
    integer i;
    
    always @(posedge i_clk) begin
        if (i_en) begin
            // 1. Valid Signal Shift
            r_valid_pipe <= {r_valid_pipe[TOTAL_PIPE_DELAY-2:0], i_valid_trigger};
            
            // 2. Address Signal Shift (데이터와 똑같이 이동)
            r_addr_pipe[0] <= i_addr; 
            for (i=1; i<TOTAL_PIPE_DELAY; i=i+1) begin
                r_addr_pipe[i] <= r_addr_pipe[i-1];
            end
        end
    end

    // =============================================================
    // 3. Calculation Pipeline (64 Parallel Lanes)
    // =============================================================
    wire signed [15:0] w_final [0:63]; 

    generate
        for (g=0; g<64; g=g+1) begin : calc_lane
            // -----------------------------------------------------
            // [Step A] Input Register & Subtraction (Latency = 1)
            // -----------------------------------------------------
            reg signed [16:0] r_sub_res; 
            reg signed [15:0] r_gamma_d1;
            reg signed [15:0] r_beta_d1;

            always @(posedge i_clk) begin
                if (i_en) begin
                    r_sub_res  <= w_raw[g] - i_mean[15:0]; 
                    r_gamma_d1 <= w_gamma[g];
                    r_beta_d1  <= w_beta[g];
                end
            end

            // -----------------------------------------------------
            // [Step B] Mult 1: (x-u) * inv_sqrt (Latency = 5)
            // -----------------------------------------------------
            wire signed [33:0] w_mult1_out; 
            
            mult_norm_ip u_mult_norm (
                .CLK(i_clk), .CE(i_en),
                .A(r_sub_res),  
                .B(i_inv_sqrt), 
                .P(w_mult1_out) 
            );

            // Gamma, Beta Delay Line (Mult1 Latency=5 만큼 지연)
            reg signed [15:0] r_gamma_delay1 [0:LAT_MULT1-1];
            reg signed [15:0] r_beta_delay1  [0:LAT_MULT1-1];
            integer k1;

            always @(posedge i_clk) begin
                if (i_en) begin
                    r_gamma_delay1[0] <= r_gamma_d1;
                    r_beta_delay1[0]  <= r_beta_d1;
                    for(k1=1; k1<LAT_MULT1; k1=k1+1) begin
                        r_gamma_delay1[k1] <= r_gamma_delay1[k1-1];
                        r_beta_delay1[k1]  <= r_beta_delay1[k1-1];
                    end
                end
            end

            // -----------------------------------------------------
            // [Step C] Mult 2: (Mult1_Res * Gamma) (Latency = 5)
            // -----------------------------------------------------
            
            // [수정] Step C에서 Shift(>>> 10)를 제거하고 Q10 포맷을 유지합니다.
            // w_mult1_out은 34비트지만 Q20 (10+10) 포맷입니다. 
            // 정밀도 유지를 위해 필요한 부분(하위 24비트)만 잘라서 입력합니다.
            wire signed [23:0] w_mult1_truncated;
            assign w_mult1_truncated = w_mult1_out[23:0]; // Q10

            wire signed [39:0] w_mult2_out; // 결과는 Q20 (Q10 * Q10)

            mult_affine_ip u_mult_affine (
                .CLK(i_clk), .CE(i_en),
                .A(w_mult1_truncated),      // Q10
                .B(r_gamma_delay1[LAT_MULT1-1]), // Q10 Gamma
                .P(w_mult2_out)             // Q20
            );

            // Beta Delay Line (Mult2 Latency=5 만큼 지연)
            reg signed [15:0] r_beta_delay2 [0:LAT_MULT2-1];
            integer k2;

            always @(posedge i_clk) begin
                if (i_en) begin
                    r_beta_delay2[0] <= r_beta_delay1[LAT_MULT1-1];
                    for(k2=1; k2<LAT_MULT2; k2=k2+1) begin
                        r_beta_delay2[k2] <= r_beta_delay2[k2-1];
                    end
                end
            end

            // -----------------------------------------------------
            // [Step D] Shift -> Add Beta -> Saturate (최종 수정)
            // -----------------------------------------------------
            reg signed [15:0] r_final_saturated;
            
            // 1. Shift (40비트 유지)
            wire signed [39:0] w_mult2_shifted;
            assign w_mult2_shifted = w_mult2_out >>> 10; // 원래대로 10 유지

            // 2. Add Beta (넓은 그릇에서 계산!)
            // [수정] 17비트로 자르지 않고, 40비트 전체를 사용하여 더합니다.
            wire signed [39:0] w_add_beta_large;
            assign w_add_beta_large = w_mult2_shifted + r_beta_delay2[LAT_MULT2-1];

            // 3. Saturate (넓은 그릇을 보고 판단)
            always @(*) begin
                // 16비트 최대값(32767)보다 크면 고정
                if (w_add_beta_large > 32767) 
                    r_final_saturated = 16'sd32767;
                // 16비트 최소값(-32768)보다 작으면 고정
                else if (w_add_beta_large < -32768) 
                    r_final_saturated = -16'sd32768;
                // 범위 안이면 하위 16비트만 통과
                else 
                    r_final_saturated = w_add_beta_large[15:0];
            end
            
            assign w_final[g] = r_final_saturated;
        end
    endgenerate

    // =============================================================
    // 4. Output Packing & Address Output
    // =============================================================
    integer k;
    always @(posedge i_clk) begin
        if (i_en) begin
            // 파이프라인 끝까지 도달한 Valid 신호 출력
            o_res_valid <= r_valid_pipe[TOTAL_PIPE_DELAY-1];
            
            // [중요] 파이프라인 끝까지 도달한 주소(Address) 출력
            o_res_addr  <= r_addr_pipe[TOTAL_PIPE_DELAY-1];

            // 데이터 Packing
            for (k=0; k<64; k=k+1) begin
                o_res_data_flat[16*k +: 16] <= w_final[k];
            end
        end else begin
            o_res_valid <= 0;
            o_res_addr  <= 0;
        end
    end

endmodule