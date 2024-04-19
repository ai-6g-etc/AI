module ahb_ai_interface #(
    parameter IDIM = 512,
    parameter NUM_HEADS = 8,
    parameter WIDTH = 8,
    parameter HIDDEN_DIM = 2048
)(
    input wire clk,
    input wire reset_n,
    output wire [31:0] ai_output,
    output wire ai_valid,
    output wire ai_ready,
    input wire ahb_hclk,
    input wire ahb_hresetn,
    input wire [31:0] ahb_haddr,
    input wire [2:0] ahb_hburst,
    input wire [3:0] ahb_hprot,
    input wire ahb_hsel,
    input wire [1:0] ahb_htrans,
    input wire [31:0] ahb_hwdata,
    input wire ahb_hwrite,
    output wire [31:0] ahb_hrdata,
    output wire ahb_hready,
    output wire ahb_hresp
);

    wire start;
    wire [IDIM*WIDTH-1:0] input_data;
    wire [IDIM*WIDTH-1:0] encoder_output;
    wire [IDIM*WIDTH-1:0] mask;
    wire [IDIM*WIDTH-1:0] output_data;
    wire done;

    reg [IDIM*HIDDEN_DIM*WIDTH-1:0] weights1;
    reg [HIDDEN_DIM*WIDTH-1:0] bias1;
    reg [HIDDEN_DIM*IDIM*WIDTH-1:0] weights2;
    reg [IDIM*WIDTH-1:0] bias2;

    // AHB interface logic
    always @(posedge ahb_hclk or negedge ahb_hresetn) begin
        if (!ahb_hresetn) begin
            weights1 <= 0;
            bias1 <= 0;
            weights2 <= 0;
            bias2 <= 0;
            start <= 0;
        end else begin
            if (ahb_hsel && ahb_hwrite) begin
                case (ahb_haddr)
                    'h0000: weights1 <= ahb_hwdata;
                    'h0004: bias1 <= ahb_hwdata;
                    'h0008: weights2 <= ahb_hwdata;
                    'h000C: bias2 <= ahb_hwdata;
                    'h0010: start <= 1'b1;
                    default: ;
                endcase
            end
            if (done) begin
                start <= 1'b0;
            end
        end
    end

    assign ahb_hrdata = done ? output_data[31:0] : 32'h0;
    assign ahb_hready = 1'b1;
    assign ahb_hresp = 1'b0; // OKAY response

    assign input_data = {IDIM{ahb_haddr[31:24]}};
    assign encoder_output = {IDIM{ahb_haddr[23:16]}};
    assign mask = {IDIM{ahb_haddr[15:8]}};

    transformer_top #(
        .IDIM(IDIM),
        .NUM_HEADS(NUM_HEADS),
        .WIDTH(WIDTH),
        .HIDDEN_DIM(HIDDEN_DIM)
    ) transformer_inst (
        .clk(clk),
        .rst_n(reset_n),
        .start(start),
        .input_data(input_data),
        .encoder_output(encoder_output),
        .mask(mask),
        .output_data(output_data),
        .done(done),
        .weights1(weights1),
        .bias1(bias1),
        .weights2(weights2),
        .bias2(bias2)
    );

    assign ai_input = ahb_haddr; // Use the AHB address as the AI input
    assign ai_valid = 1'b1; // For simplicity, we assume the AI network is always valid
    assign ai_ready = 1'b1; // For simplicity, we assume the AI network is always ready

endmodule
