// total latency : 7

module partial_sum_64 (
    input  wire          i_clk,
    input  wire          i_rst,
    input  wire          i_en,
    input  wire [1023:0] i_data_flat, // 16bit * 64 data

    output reg signed [21:0] o_part_sum,    // Final Sum
    output reg signed [37:0] o_part_sq_sum  // Final Square Sum
);

    // 1. Data Unpacking & Stage 0 (Square)
    integer k;
    reg signed [15:0] x [0:63];     // Input registers
    reg signed [31:0] x_sq [0:63];  // Square registers

    // 2. Adder Tree Registers (Fully Pipelined)
    
    // Sum path
    reg signed [16:0] sum_stg1 [0:31];
    reg signed [17:0] sum_stg2 [0:15];
    reg signed [18:0] sum_stg3 [0:7];
    reg signed [19:0] sum_stg4 [0:3];
    reg signed [20:0] sum_stg5 [0:1];
    
    // Square Sum path
    reg signed [32:0] sq_stg1 [0:31];
    reg signed [33:0] sq_stg2 [0:15];
    reg signed [34:0] sq_stg3 [0:7];
    reg signed [35:0] sq_stg4 [0:3];
    reg signed [36:0] sq_stg5 [0:1];

    integer i;

    always @(posedge i_clk) begin
        if (i_rst) begin
            o_part_sum    <= 0;
            o_part_sq_sum <= 0;
            
            for (k=0; k<64; k=k+1) begin
                x[k] <= 0; x_sq[k] <= 0;
            end
            for (k=0; k<32; k=k+1) begin
                sum_stg1[k] <= 0; sq_stg1[k] <= 0;
            end
            for (k=0; k<16; k=k+1) begin
                sum_stg2[k] <= 0; sq_stg2[k] <= 0;
            end
            for (k=0; k<8; k=k+1) begin
                sum_stg3[k] <= 0; sq_stg3[k] <= 0;
            end
            for (k=0; k<4; k=k+1) begin
                sum_stg4[k] <= 0; sq_stg4[k] <= 0;
            end
            for (k=0; k<2; k=k+1) begin
                sum_stg5[k] <= 0; sq_stg5[k] <= 0;
            end

        end else if (i_en) begin
            // Unpack & Square (Stage 0)
            for (k=0; k<64; k=k+1) begin
                x[k]    <= $signed(i_data_flat[16*k +: 16]); 
                x_sq[k] <= $signed(i_data_flat[16*k +: 16]) * $signed(i_data_flat[16*k +: 16]); 
            end
            
            // Stage 1 (64 -> 32)
            for (i=0; i<32; i=i+1) begin
                sum_stg1[i] <= x[2*i]    + x[2*i+1];
                sq_stg1[i]  <= x_sq[2*i] + x_sq[2*i+1];
            end

            // Stage 2 (32 -> 16)
            for (i=0; i<16; i=i+1) begin
                sum_stg2[i] <= sum_stg1[2*i] + sum_stg1[2*i+1];
                sq_stg2[i]  <= sq_stg1[2*i] + sq_stg1[2*i+1];
            end

            // Stage 3 (16 -> 8)
            for (i=0; i<8; i=i+1) begin
                sum_stg3[i] <= sum_stg2[2*i] + sum_stg2[2*i+1];
                sq_stg3[i]  <= sq_stg2[2*i] + sq_stg2[2*i+1];
            end

            // Stage 4 (8 -> 4)
            for (i=0; i<4; i=i+1) begin
                sum_stg4[i] <= sum_stg3[2*i] + sum_stg3[2*i+1];
                sq_stg4[i]  <= sq_stg3[2*i] + sq_stg3[2*i+1];
            end

            // Stage 5 (4 -> 2)
            for (i=0; i<2; i=i+1) begin
                sum_stg5[i] <= sum_stg4[2*i] + sum_stg4[2*i+1];
                sq_stg5[i]  <= sq_stg4[2*i] + sq_stg4[2*i+1];
            end

            // Stage 6 (2 -> 1) : Final Output
            o_part_sum    <= sum_stg5[0] + sum_stg5[1]; 
            o_part_sq_sum <= sq_stg5[0] + sq_stg5[1]; 
        end
    end

endmodule