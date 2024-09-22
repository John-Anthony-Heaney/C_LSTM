#include <stdio.h>
#include <math.h>

// Sigmoid function
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Tanh function
float tanh_activation(float x) {
    return tanh(x);
}

// Matrix-vector multiplication function
void matvec_mul(float* matrix, float* vector, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0;  // Initialize result element
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

// Forget gate function
void forget_gate(float* W_f, float* U_f, float* x_t, float* h_prev, float* b_f, float* f_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_f, x_t, temp, hidden_size, input_size);
    matvec_mul(U_f, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        f_t[i] = sigmoid(temp[i] + temp_h[i] + b_f[i]);
    }
}

// Input gate function
void input_gate(float* W_i, float* U_i, float* x_t, float* h_prev, float* b_i, float* i_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_i, x_t, temp, hidden_size, input_size);
    matvec_mul(U_i, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        i_t[i] = sigmoid(temp[i] + temp_h[i] + b_i[i]);
    }
}

// Cell state update function
void cell_update(float* W_c, float* U_c, float* x_t, float* h_prev, float* b_c, float* c_hat_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_c, x_t, temp, hidden_size, input_size);
    matvec_mul(U_c, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        c_hat_t[i] = tanh_activation(temp[i] + temp_h[i] + b_c[i]);
    }
}

// Output gate function
void output_gate(float* W_o, float* U_o, float* x_t, float* h_prev, float* b_o, float* o_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_o, x_t, temp, hidden_size, input_size);
    matvec_mul(U_o, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        o_t[i] = sigmoid(temp[i] + temp_h[i] + b_o[i]);
    }
}

// Function to update the cell state
void update_cell_state(float* f_t, float* i_t, float* c_prev, float* c_hat_t, float* c_t, int hidden_size) {
    for (int i = 0; i < hidden_size; i++) {
        c_t[i] = f_t[i] * c_prev[i] + i_t[i] * c_hat_t[i];
    }
}

// Function to update the hidden state
void update_hidden_state(float* o_t, float* c_t, float* h_t, int hidden_size) {
    for (int i = 0; i < hidden_size; i++) {
        h_t[i] = o_t[i] * tanh_activation(c_t[i]);
    }
}

// Full LSTM cell function (one time step)
void lstm_cell(float* W_f, float* U_f, float* b_f,
               float* W_i, float* U_i, float* b_i,
               float* W_c, float* U_c, float* b_c,
               float* W_o, float* U_o, float* b_o,
               float* x_t, float* h_prev, float* c_prev,
               float* h_t, float* c_t,
               int input_size, int hidden_size) {

    float f_t[hidden_size], i_t[hidden_size], c_hat_t[hidden_size], o_t[hidden_size];

    // Step 1: Forget gate
    forget_gate(W_f, U_f, x_t, h_prev, b_f, f_t, input_size, hidden_size);

    // Step 2: Input gate
    input_gate(W_i, U_i, x_t, h_prev, b_i, i_t, input_size, hidden_size);

    // Step 3: Cell state update (candidate)
    cell_update(W_c, U_c, x_t, h_prev, b_c, c_hat_t, input_size, hidden_size);

    // Step 4: Update cell state
    update_cell_state(f_t, i_t, c_prev, c_hat_t, c_t, hidden_size);

    // Step 5: Output gate
    output_gate(W_o, U_o, x_t, h_prev, b_o, o_t, input_size, hidden_size);

    // Step 6: Update hidden state
    update_hidden_state(o_t, c_t, h_t, hidden_size);
}

int main() {
    // Example: LSTM cell for one time step

    // Define the size of the input and hidden states
    int input_size = 3;   // Example: 3 input features
    int hidden_size = 2;  // Example: 2 hidden units

    // Define input x_t (current input) and previous states (h_prev, c_prev)
    float x_t[] = {1.0, 0.5, -1.2};    // Input vector (size 3)
    float h_prev[] = {0.1, -0.2};      // Previous hidden state (size 2)
    float c_prev[] = {0.2, -0.1};      // Previous cell state (size 2)

    // Define weight matrices and biases for each gate
    // Forget gate parameters
    float W_f[] = {0.5, 0.8, -0.3,   // Weight matrix for input (size 2x3)
                   -0.1, 0.4, 0.7};
    float U_f[] = {0.2, 0.6,         // Weight matrix for hidden state (size 2x2)
                   -0.5, 0.9};
    float b_f[] = {0.05, -0.1};      // Bias for forget gate

    // Input gate parameters
    float W_i[] = {0.4, 0.9, -0.2,   // Weight matrix for input (size 2x3)
                   0.3, -0.5, 0.7};
    float U_i[] = {0.1, 0.8,         // Weight matrix for hidden state (size 2x2)
                   -0.6, 0.4};
    float b_i[] = {0.02, -0.05};     // Bias for input gate

    // Cell state update parameters
    float W_c[] = {0.3, -0.1, 0.2,   // Weight matrix for input (size 2x3)
                   -0.4, 0.6, -0.7};
    float U_c[] = {0.5, -0.3,        // Weight matrix for hidden state (size 2x2)
                   0.4, -0.2};
    float b_c[] = {0.01, 0.02};      // Bias for cell state update

    // Output gate parameters
    float W_o[] = {0.6, -0.7, 0.4,   // Weight matrix for input (size 2x3)
                   0.5, -0.2, 0.8};
    float U_o[] = {0.3, 0.9,         // Weight matrix for hidden state (size 2x2)
                   -0.4, 0.2};
    float b_o[] = {0.04, -0.06};     // Bias for output gate

    // Define arrays to store the updated hidden state and cell state
    float h_t[hidden_size];
    float c_t[hidden_size];

    // Call the full LSTM cell function
    lstm_cell(W_f, U_f, b_f, W_i, U_i, b_i, W_c, U_c, b_c, W_o, U_o, b_o,
              x_t, h_prev, c_prev, h_t, c_t, input_size, hidden_size);

    // Print the updated hidden state and cell state
    printf("Updated hidden state:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("h_t[%d] = %.5f\n", i, h_t[i]);
    }

    printf("\nUpdated cell state:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("c_t[%d] = %.5f\n", i, c_t[i]);
    }

    return 0;
}
