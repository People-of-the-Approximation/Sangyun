`timescale 1ns / 1ps

module ln_stage3_normalize (
    // 순수 Combinational Logic + 1 Clock Register 형태가 일반적이지만
    // 원본 구조 유지를 위해 Clocked Logic으로 구현
    
    input  wire i_clk,
    input  wire i_en,
    input  wire i_valid_trigger, // 정규화 수행 신호
    
    // 파라미터 및 통계치
    input  wire signed [31:0] i_mean,
    input  wire signed [16:0] i_inv_sqrt, // 부호비트 포함 확장됨
    
    // 벡터 데이터 (입력)
    input  wire [1023:0] i_raw_data_flat,
    input  wire [1023:0] i_gamma_flat, // Flattened Gamma (64x16)
    input  wire [1023:0] i_beta_flat,  // Flattened Beta (64x16)

    // 벡터 데이터 (출력)
    output reg  [1023:0] o_res_data_flat,
    output reg           o_res_valid
);

    integer k;
    reg signed [15:0] r_raw_val;
    reg signed [31:0] r_norm_temp;  
    reg signed [31:0] r_final_val;  
    reg signed [15:0] w_gamma;
    reg signed [15:0] w_beta;

    always @(posedge i_clk) begin
        if (i_en && i_valid_trigger) begin
            for (k=0; k<64; k=k+1) begin
                // 1. Slicing
                r_raw_val = i_raw_data_flat[16*k +: 16];
                w_gamma   = i_gamma_flat[16*k +: 16];
                w_beta    = i_beta_flat[16*k +: 16];

                // 2. Normalization: (x - u) * inv_sqrt
                r_norm_temp = (r_raw_val - i_mean[15:0]) * i_inv_sqrt;
                
                // 3. Affine: * gamma + beta
                r_final_val = ((r_norm_temp >>> 10) * w_gamma) + w_beta; 

                // 4. Output Packing
                o_res_data_flat[16*k +: 16] <= r_final_val[15:0]; 
            end
            o_res_valid <= 1'b1;
        end else begin
            o_res_valid <= 1'b0;
        end
    end

endmodule