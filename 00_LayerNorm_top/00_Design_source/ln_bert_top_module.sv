module ln_bert_top_module (
    input  wire          i_clk, i_rst, i_en,
    input  wire          i_valid,
    input  wire [1023:0] i_data_flat,
    output reg  [1023:0] o_result_flat,
    output reg           o_valid
);

    // =============================================================
    // 1. Scheduler (12 Cycle Fixed) - 이제 병목 없음!
    // =============================================================
    reg [3:0] cycle_cnt; 
    reg [1:0] ptr_in;
    
    always @(posedge i_clk) begin
        if (i_rst) begin
            cycle_cnt <= 0; ptr_in <= 0;
        end else if (i_en && i_valid) begin
            if (cycle_cnt == 11) begin
                cycle_cnt <= 0;
                ptr_in    <= ptr_in + 1;
            end else begin
                cycle_cnt <= cycle_cnt + 1;
            end
        end
    end

    // =============================================================
    // 2. Memory & Buffers
    // =============================================================
    (* ram_style = "distributed" *) reg [1023:0] input_bram [0:3][0:11];
    (* ram_style = "distributed" *) reg [1023:0] output_bram [0:3][0:11];
    reg signed [30:0] bank_sum [0:3];
    reg signed [50:0] bank_sq_sum [0:3];
    (* ram_style = "distributed" *) reg [1023:0] rom_gamma_flat [0:11];
    (* ram_style = "distributed" *) reg [1023:0] rom_beta_flat [0:11];

    // =============================================================
    // 3. Stage 1: Accumulate (Cycle 0~11)
    // =============================================================
    wire s1_valid;
    wire [1:0] s1_ptr;
    wire [3:0] s1_cnt;
    wire signed [21:0] s1_sum;
    wire signed [37:0] s1_sq_sum;

    ln_stage1_accumulate u_stage1 (
        .i_clk(i_clk), .i_rst(i_rst), .i_en(i_en),
        .i_valid(i_valid), .i_data_flat(i_data_flat),
        .i_ptr_in(ptr_in), .i_cycle_cnt(cycle_cnt),
        .o_acc_valid(s1_valid), .o_acc_ptr(s1_ptr), .o_acc_cnt(s1_cnt),
        .o_part_sum(s1_sum), .o_part_sq_sum(s1_sq_sum)
    );

    always @(posedge i_clk) begin
        if (i_en && i_valid) input_bram[ptr_in][cycle_cnt] <= i_data_flat;
        if (i_en && s1_valid) begin
            if (s1_cnt == 0) begin
                bank_sum[s1_ptr] <= s1_sum; bank_sq_sum[s1_ptr] <= s1_sq_sum;
            end else begin
                bank_sum[s1_ptr] <= bank_sum[s1_ptr] + s1_sum;
                bank_sq_sum[s1_ptr] <= bank_sq_sum[s1_ptr] + s1_sq_sum;
            end
        end
    end

    // =============================================================
    // 4. Stage 2: Variance Calc (New Stage)
    // =============================================================
    reg is_warmed_up;
    always @(posedge i_clk) begin
        if (i_rst) is_warmed_up <= 0;
        else if (ptr_in == 1) is_warmed_up <= 1; 
    end

    // [2] s2_start 조건 수정 (반드시 수정!)
    // ptr_in이 다시 0으로 돌아왔을 때 죽지 않도록 방어 코드 추가
    wire s2_start = (cycle_cnt == 7) && ( (ptr_in != 0) || is_warmed_up ) && i_en;
    wire [1:0] s2_target_bank = ptr_in - 1;

    wire signed [31:0] s2_mean;
    wire [15:0]        s2_var;
    wire               s2_valid;
    wire [1:0]         s2_bank;

    ln_stage2_calc_var u_stage2 (
        .i_clk(i_clk), .i_en(i_en),
        .i_start(s2_start), .i_bank_id(s2_target_bank),
        .i_sum(bank_sum[s2_target_bank]), .i_sq_sum(bank_sq_sum[s2_target_bank]),
        .o_mean(s2_mean), .o_variance(s2_var), .o_valid(s2_valid), .o_bank_id(s2_bank)
    );

    // =============================================================
    // 5. Stage 3: PWL InvSqrt (New Stage)
    // =============================================================
    wire signed [31:0] s3_mean;
    wire [15:0]        s3_inv_sqrt;
    wire               s3_valid;
    wire [1:0]         s3_bank;

    ln_stage3_calc_pwl u_stage3 (
        .i_clk(i_clk), .i_rst(i_rst), .i_en(i_en),
        .i_valid(s2_valid), .i_bank_id(s2_bank), // Stage 2 출력을 입력으로
        .i_mean(s2_mean), .i_variance(s2_var),
        .o_mean(s3_mean), .o_inv_sqrt(s3_inv_sqrt), .o_valid(s3_valid), .o_bank_id(s3_bank)
    );

    // =============================================================
    // 6. Stage 4: Normalize (Old Stage 3)
    // =============================================================
    // Stage 3 완료 시 레지스터 캡처 (Timing Buffer)
    reg signed [31:0] s4_mean_reg;
    reg signed [16:0] s4_inv_sqrt_reg;
    reg [1:0]         s4_active_bank;
    
    always @(posedge i_clk) begin
        if (s3_valid) begin
            s4_mean_reg     <= s3_mean;
            s4_inv_sqrt_reg <= {1'b0, s3_inv_sqrt};
            s4_active_bank  <= s3_bank;
        end
    end

    wire [5:0] s4_in_tag = {s4_active_bank, cycle_cnt}; // Bank + Cycle
    wire [5:0] s4_out_tag;
    wire [1023:0] s4_res_data;
    wire s4_res_valid;

    ln_stage4_normalize u_stage4 (
        .i_clk(i_clk), .i_en(i_en), .i_valid_trigger(i_en),
        .i_addr(s4_in_tag),
        .i_mean(s4_mean_reg), .i_inv_sqrt(s4_inv_sqrt_reg),
        .i_raw_data_flat(input_bram[s4_active_bank][cycle_cnt]),
        .i_gamma_flat(rom_gamma_flat[cycle_cnt]), .i_beta_flat(rom_beta_flat[cycle_cnt]),
        .o_res_data_flat(s4_res_data), .o_res_valid(s4_res_valid), .o_res_addr(s4_out_tag)
    );

    // =============================================================
    // 7. Output
    // =============================================================
    always @(posedge i_clk) begin
        if (s4_res_valid) begin
            output_bram[s4_out_tag[5:4]][s4_out_tag[3:0]] <= s4_res_data;
            o_valid       <= 1;
            o_result_flat <= s4_res_data;
        end else begin
            o_valid       <= 0;
            o_result_flat <= 0;
        end
    end

    initial begin
        $readmemh("bert_gamma.mem", rom_gamma_flat);
        $readmemh("bert_beta.mem",  rom_beta_flat);
    end

endmodule