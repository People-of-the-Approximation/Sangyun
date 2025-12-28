`timescale 1ns / 1ps

module ln_stage2_calc_var (
    input  wire          i_clk,
    input  wire          i_en,
    input  wire          i_start, 
    input  wire [1:0]    i_bank_id, 
    
    input  wire signed [30:0] i_sum,
    input  wire signed [50:0] i_sq_sum,

    output reg  signed [31:0] o_mean, 
    output reg  [15:0]        o_variance,
    output reg                o_valid,
    output reg  [1:0]         o_bank_id
);

    // -------------------------------------------------------------
    // 1차 곱셈 (Mean, Ex2) - Latency 5
    // -------------------------------------------------------------
    wire signed [42:0] w_mult_mean;
    wire signed [62:0] w_mult_ex2;
    
    mult_sum_const u_mult_mean (.CLK(i_clk), .CE(i_en), .A(i_sum), .B(12'd1365), .P(w_mult_mean));
    mult_sq_sum_const u_mult_ex2 (.CLK(i_clk), .CE(i_en), .A(i_sq_sum), .B(12'd1365), .P(w_mult_ex2));

    // Delay Control (Depth 5)
    reg [4:0] r_val_d1;
    reg [1:0] r_bank_d1 [0:4];
    integer i;
    always @(posedge i_clk) begin
        if (i_en) begin
            r_val_d1 <= {r_val_d1[3:0], i_start};
            r_bank_d1[0] <= i_bank_id;
            for(i=1; i<5; i=i+1) r_bank_d1[i] <= r_bank_d1[i-1];
        end
    end

    // -------------------------------------------------------------
    // Shift (Latency 1)
    // -------------------------------------------------------------
    reg signed [31:0] r_mean_sh, r_ex2_sh;
    reg r_val_step2;
    reg [1:0] r_bank_step2;

    always @(posedge i_clk) begin
        if (i_en) begin
            r_mean_sh <= w_mult_mean >>> 20;
            r_ex2_sh  <= w_mult_ex2  >>> 20;
            r_val_step2 <= r_val_d1[4];
            r_bank_step2 <= r_bank_d1[4];
        end
    end

    // -------------------------------------------------------------
    // 2차 곱셈 (Mean^2) - Latency 5
    // -------------------------------------------------------------
    wire signed [63:0] w_mean_sq;
    mult_mean_sq u_mult_sq (.CLK(i_clk), .CE(i_en), .A(r_mean_sh), .B(r_mean_sh), .P(w_mean_sq));

    // Delay Line (Depth 5)
    reg signed [31:0] r_ex2_d2 [0:4];
    reg signed [31:0] r_mean_d2 [0:4];
    reg [4:0] r_val_d2;
    reg [1:0] r_bank_d2 [0:4];
    integer j;

    always @(posedge i_clk) begin
        if (i_en) begin
            r_ex2_d2[0]  <= r_ex2_sh;
            r_mean_d2[0] <= r_mean_sh;
            r_val_d2     <= {r_val_d2[3:0], r_val_step2};
            r_bank_d2[0] <= r_bank_step2;
            for(j=1; j<5; j=j+1) begin
                r_ex2_d2[j] <= r_ex2_d2[j-1];
                r_mean_d2[j] <= r_mean_d2[j-1];
                r_bank_d2[j] <= r_bank_d2[j-1];
            end
        end
    end

    // -------------------------------------------------------------
    // 뺄셈 (Variance) - Latency 1 (총 12클럭 지점)
    // -------------------------------------------------------------
    reg signed [31:0] raw_var;

    always @(posedge i_clk) begin
        if (i_en) begin
            if (r_val_d2[4]) begin
                o_mean    <= r_mean_d2[4];
                o_bank_id <= r_bank_d2[4]; // [확인] Bank ID 출력 (정상)
                o_valid   <= 1'b1;

                // Variance Calc & Saturate (Signed Casting 추가)
                raw_var = r_ex2_d2[4] - $signed(w_mean_sq[31:0]);
                
                if (raw_var[31])          o_variance <= 0;
                else if (|raw_var[30:16]) o_variance <= 16'hFFFF;
                else                      o_variance <= raw_var[15:0];
            end else begin
                o_valid <= 1'b0;
                // o_bank_id <= 0; // (선택사항) Valid 없을 때 0으로 밀어도 됨
            end
        end
    end

endmodule