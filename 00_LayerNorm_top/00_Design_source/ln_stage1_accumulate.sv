module ln_stage1_accumulate (
    input  wire          i_clk,
    input  wire          i_rst,
    input  wire          i_en,
    input  wire          i_valid,
    input  wire [1023:0] i_data_flat,
    
    // 외부 제어 신호 패스스루 (Latency 보상용)
    input  wire [1:0]    i_ptr_in,
    input  wire [3:0]    i_cycle_cnt,

    // 출력: 지연된 Valid 및 계산 결과
    output wire          o_acc_valid,    // BRAM/Register 누적 Enable 신호
    output wire [1:0]    o_acc_ptr,      // 지연된 포인터
    output wire [3:0]    o_acc_cnt,      // 지연된 카운터
    output wire signed [21:0] o_part_sum,
    output wire signed [37:0] o_part_sq_sum
);

    // partial_sum의 7클럭 지연
    localparam LATENCY = 7; 

    reg [LATENCY-1:0] r_valid_delay;        
    reg [1:0]         r_ptr_delay [0:LATENCY-1]; 
    reg [3:0]         r_cnt_delay [0:LATENCY-1]; 

    integer i;
    always @(posedge i_clk) begin
        if (i_en) begin
            // Shift Register
            r_valid_delay[0] <= i_valid;
            r_ptr_delay[0]   <= i_ptr_in;
            r_cnt_delay[0]   <= i_cycle_cnt;
            
            for (i=1; i<LATENCY; i=i+1) begin
                r_valid_delay[i] <= r_valid_delay[i-1];
                r_ptr_delay[i]   <= r_ptr_delay[i-1];
                r_cnt_delay[i]   <= r_cnt_delay[i-1];
            end
        end
    end

    assign o_acc_valid = r_valid_delay[LATENCY-1];
    assign o_acc_ptr   = r_ptr_delay[LATENCY-1];
    assign o_acc_cnt   = r_cnt_delay[LATENCY-1];

    partial_sum_64 u_adder (
        .i_clk(i_clk), 
        .i_rst(i_rst), 
        .i_en(i_en),
        .i_data_flat(i_data_flat),
        .o_part_sum(o_part_sum),
        .o_part_sq_sum(o_part_sq_sum)
    );

endmodule