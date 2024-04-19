`timescale 1ns/1ps

module multi_head_attention #(
    parameter IDIM = 512,    
    parameter NUM_HEADS = 8, 
    parameter HEAD_DIM = IDIM / NUM_HEADS,
    parameter WIDTH = 8     
)(
    input clk,
    input rst_n,
    input start,
    input [IDIM*WIDTH-1:0] input_q,  
    input [IDIM*WIDTH-1:0] input_k,  
    input [IDIM*WIDTH-1:0] input_v, 
    input [IDIM*WIDTH-1:0] mask,    
    output reg [IDIM*WIDTH-1:0] output_data,
    output reg done
);


    reg [HEAD_DIM*WIDTH-1:0] q[NUM_HEADS];
    reg [HEAD_DIM*WIDTH-1:0] k[NUM_HEADS];
    reg [HEAD_DIM*WIDTH-1:0] v[NUM_HEADS];
    reg [HEAD_DIM*WIDTH-1:0] scaled_dot_product[NUM_HEADS];
    reg [HEAD_DIM*WIDTH-1:0] attn_output[NUM_HEADS];


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data <= 0;
            done <= 0;
        end else if (start) begin
            for (int i = 0; i < NUM_HEADS; i = i + 1) begin
                q[i] <= input_q[(i+1)*HEAD_DIM*WIDTH-1 -: HEAD_DIM*WIDTH];
                k[i] <= input_k[(i+1)*HEAD_DIM*WIDTH-1 -: HEAD_DIM*WIDTH];
                v[i] <= input_v[(i+1)*HEAD_DIM*WIDTH-1 -: HEAD_DIM*WIDTH];
            end
           
            for (int i = 0; i < NUM_HEADS; i = i + 1) begin
                compute_scaled_dot_product(q[i], k[i], scaled_dot_product[i], mask);
                compute_attn_output(scaled_dot_product[i], v[i], attn_output[i]);
            end
            done <= 1;
        end
    end


    task automatic compute_scaled_dot_product;
        input [HEAD_DIM*WIDTH-1:0] query;
        input [HEAD_DIM*WIDTH-1:0] key;
        output reg [HEAD_DIM*WIDTH-1:0] result;
        input [IDIM*WIDTH-1:0] mask;
        begin
            reg [2*HEAD_DIM*WIDTH-1:0] dot_product;
            dot_product = 0;
            for (int i = 0; i < HEAD_DIM; i = i + 1) begin
                dot_product = dot_product + query[i*WIDTH +: WIDTH] * key[i*WIDTH +: WIDTH];
            end
            dot_product = dot_product / $sqrt(HEAD_DIM);
            result = dot_product;
        end
    endtask


    task automatic compute_attn_output;
        input [HEAD_DIM*WIDTH-1:0] scaled_dot_product;
        input [HEAD_DIM*WIDTH-1:0] value;
        output reg [HEAD_DIM*WIDTH-1:0] result;
        begin
            result = 0;
            for (int i = 0; i < HEAD_DIM; i = i + 1) begin
                result = result + scaled_dot_product[i*WIDTH +: WIDTH] * value[i*WIDTH +: WIDTH];
            end
        end
    endtask


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data <= 0;
        end else if (done) begin
            output_data <= 0;
            for (int i = 0; i < NUM_HEADS; i = i + 1) begin
                output_data <= output_data + attn_output[i];
            end
        end
    end

endmodule

module feedforward_network #(
    parameter IDIM = 512,  
    parameter WIDTH = 8    
)(
    input clk,
    input rst_n,
    input start,
    input [IDIM*WIDTH-1:0] input_data,
    output reg [IDIM*WIDTH-1:0] output_data,
    output reg done
);

    reg [IDIM*WIDTH-1:0] hidden_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hidden_state <= 0;
            output_data <= 0;
            done <= 0;
        end else if (start) begin
            
            hidden_state <= input_data * weights1 + bias1;
          
            hidden_state <= relu(hidden_state);
          
            output_data <= hidden_state * weights2 + bias2;
            done <= 1;
        end
    end

    function [IDIM*WIDTH-1:0] relu;
        input [IDIM*WIDTH-1:0] x;
        begin
            relu = (x > 0) ? x : 0;
        end
    endfunction

endmodule

module encoder_layer #(
    parameter IDIM = 512,    
    parameter NUM_HEADS = 8,
    parameter WIDTH = 8      
)(
    input clk,
    input rst_n,
    input start,
    input [IDIM*WIDTH-1:0] input_data,
    input [IDIM*WIDTH-1:0] mask,
    output reg [IDIM*WIDTH-1:0] output_data,
    output reg done
);

    wire [IDIM*WIDTH-1:0] mha_output;
    wire mha_done;
    wire [IDIM*WIDTH-1:0] ffn_output;
    wire ffn_done;

    multi_head_attention #(
        .IDIM(IDIM),
        .NUM_HEADS(NUM_HEADS),
        .WIDTH(WIDTH)
    ) mha_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .input_q(input_data),
        .input_k(input_data),
        .input_v(input_data),
        .mask(mask),
        .output_data(mha_output),
        .done(mha_done)
    );

    feedforward_network #(
        .IDIM(IDIM),
        .WIDTH(WIDTH)
    ) ffn_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(mha_done),
        .input_data(mha_output),
        .output_data(ffn_output),
        .done(ffn_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data <= 0;
            done <= 0;
        end else if (ffn_done) begin
            output_data <= ffn_output;
            done <= 1;
        end
    end

endmodule

module decoder_layer #(
    parameter IDIM = 512,    
    parameter NUM_HEADS = 8, 
    parameter WIDTH = 8      
)(
    input clk,
    input rst_n,
    input start,
    input [IDIM*WIDTH-1:0] input_data,
    input [IDIM*WIDTH-1:0] encoder_output,
    input [IDIM*WIDTH-1:0] mask,
    output reg [IDIM*WIDTH-1:0] output_data,
    output reg done
);

    wire [IDIM*WIDTH-1:0] self_attn_output;
    wire self_attn_done;
    wire [IDIM*WIDTH-1:0] cross_attn_output;
    wire cross_attn_done;
    wire [IDIM*WIDTH-1:0] ffn_output;
    wire ffn_done;

    multi_head_attention #(
        .IDIM(IDIM),
        .NUM_HEADS(NUM_HEADS),
        .WIDTH(WIDTH)
    ) self_attn_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .input_q(input_data),
        .input_k(input_data),
        .input_v(input_data),
        .mask(mask),
        .output_data(self_attn_output),
        .done(self_attn_done)
    );

    multi_head_attention #(
        .IDIM(IDIM),
        .NUM_HEADS(NUM_HEADS),
        .WIDTH(WIDTH)
    ) cross_attn_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(self_attn_done),
        .input_q(self_attn_output),
        .input_k(encoder_output),
        .input_v(encoder_output),
        .mask(mask),
        .output_data(cross_attn_output),
        .done(cross_attn_done)
    );

    feedforward_network #(
        .IDIM(IDIM),
        .WIDTH(WIDTH)
    ) ffn_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(cross_attn_done),
        .input_data(cross_attn_output),
        .output_data(ffn_output),
        .done(ffn_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data <= 0;
            done <= 0;
        end else if (ffn_done) begin
            output_data <= ffn_output;
            done <= 1;
        end
    end

endmodule

module transformer_top #(
    parameter IDIM = 512,    
    parameter NUM_HEADS = 8, 
    parameter WIDTH = 8     
)(
    input clk,
    input rst_n,
    input start,
    input [IDIM*WIDTH-1:0] input_data,
    input [IDIM*WIDTH-1:0] encoder_output,
    input [IDIM*WIDTH-1:0] mask,
    output reg [IDIM*WIDTH-1:0] output_data,
    output reg done
);

    wire [IDIM*WIDTH-1:0] encoder_layer_output;
    wire encoder_layer_done;
    wire [IDIM*WIDTH-1:0] decoder_layer_output;
    wire decoder_layer_done;

    encoder_layer #(
        .IDIM(IDIM),
        .NUM_HEADS(NUM_HEADS),
        .WIDTH(WIDTH)
    ) encoder_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .input_data(input_data),
        .mask(mask),
        .output_data(encoder_layer_output),
        .done(encoder_layer_done)
    );

    decoder_layer #(
        .IDIM(IDIM),
        .NUM_HEADS(NUM_HEADS),
        .WIDTH(WIDTH)
    ) decoder_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(encoder_layer_done),
        .input_data(input_data),
        .encoder_output(encoder_output),
        .mask(mask),
        .output_data(decoder_layer_output),
        .done(decoder_layer_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data <= 0;
            done <= 0;
        end else if (decoder_layer_done) begin
            output_data <= decoder_layer_output;
            done <= 1;
        end
    end

endmodule
