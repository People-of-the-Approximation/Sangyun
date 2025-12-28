`timescale 1ns / 1ps

module ln_stage3_calc_pwl (
    input  wire          i_clk,
    input  wire          i_rst,
    input  wire          i_en,
    input  wire          i_valid,
    input  wire [1:0]    i_bank_id,
    
    input  wire signed [31:0] i_mean,
    input  wire [15:0]        i_variance,

    output wire signed [31:0] o_mean,
    
    // [수정] 바로 내보내지 않고 reg로 잡아서 1클럭 지연
    output reg  [15:0]        o_inv_sqrt, 
    output reg                o_valid,    
    output wire [1:0]         o_bank_id
);

    localparam PWL_LATENCY = 12;

    // 1. PWL (Inverse Sqrt) - Raw Output (11클럭 만에 나옴)
    wire [15:0] w_inv_sqrt_raw;
    wire        w_valid_raw;

    pwl_approx u_pwl (
        .i_clk(i_clk), .i_rst(i_rst), .i_en(i_en),
        .i_valid(i_valid),
        .i_variance(i_variance),
        .o_result(w_inv_sqrt_raw),
        .o_valid(w_valid_raw)
    );

    // [수정 핵심] 11클럭 + 1클럭(Reg) = 12클럭 완성
    always @(posedge i_clk) begin
        if (i_en) begin
            o_inv_sqrt <= w_inv_sqrt_raw >>> 6 ;
            o_valid    <= w_valid_raw;
        end
    end

    // 2. Mean & Bank ID Sync (PWL Latency 12만큼 지연)
    // 이쪽은 Shift Register가 12칸이라 이미 12클럭임. 그대로 둠.
    reg signed [31:0] r_mean_sync [0:PWL_LATENCY-1];
    reg [1:0]         r_bank_sync [0:PWL_LATENCY-1];
    integer i;

    always @(posedge i_clk) begin
        if (i_en) begin
            r_mean_sync[0] <= i_mean;
            r_bank_sync[0] <= i_bank_id;
            for (i=1; i<PWL_LATENCY; i=i+1) begin
                r_mean_sync[i] <= r_mean_sync[i-1];
                r_bank_sync[i] <= r_bank_sync[i-1];
            end
        end
    end

    assign o_mean    = r_mean_sync[PWL_LATENCY-1];
    assign o_bank_id = r_bank_sync[PWL_LATENCY-1];

endmodule