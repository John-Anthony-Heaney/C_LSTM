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

// Function to multiply a matrix with a vector
void matvec_mul(float* matrix, float* vector, float* result, int rows, int cols) {
    // Loop through rows of the matrix
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

    // Multiply W_f with the input x_t
    matvec_mul(W_f, x_t, temp, hidden_size, input_size);
    
    // Multiply U_f with the previous hidden state h_prev
    matvec_mul(U_f, h_prev, temp_h, hidden_size, hidden_size);
    
    // Add the results together and apply the bias, then apply the sigmoid function
    for (int i = 0; i < hidden_size; i++) {
        f_t[i] = sigmoid(temp[i] + temp_h[i] + b_f[i]);
    }
}


// Input gate function
void input_gate(float* W_i, float* U_i, float* x_t, float* h_prev, float* b_i, float* i_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    // Multiply W_i with the input x_t
    matvec_mul(W_i, x_t, temp, hidden_size, input_size);
    
    // Multiply U_i with the previous hidden state h_prev
    matvec_mul(U_i, h_prev, temp_h, hidden_size, hidden_size);
    
    // Add the results together and apply the bias, then apply the sigmoid function
    for (int i = 0; i < hidden_size; i++) {
        i_t[i] = sigmoid(temp[i] + temp_h[i] + b_i[i]);
    }
}

// Function to update the cell state
void update_cell_state(float* f_t, float* i_t, float* c_prev, float* c_hat_t, float* c_t, int hidden_size) {
    for (int i = 0; i < hidden_size; i++) {
        // Update the cell state: C_t = f_t * C_{t-1} + i_t * c_hat_t
        c_t[i] = f_t[i] * c_prev[i] + i_t[i] * c_hat_t[i];
    }
}

int main() {
     // Example: Updating the cell state for one time step

    // Define the size of the hidden states
    int hidden_size = 2;  // Example: 2 hidden units

    // Define the forget gate output f_t, input gate output i_t, previous cell state C_{t-1}, and candidate cell state c_hat_t
    float f_t[] = {0.7, 0.5};           // Forget gate output (size 2)
    float i_t[] = {0.6, 0.4};           // Input gate output (size 2)
    float c_prev[] = {0.2, -0.1};       // Previous cell state (size 2)
    float c_hat_t[] = {0.3, -0.5};      // Candidate cell state (size 2)

    // Define an array to store the updated cell state
    float c_t[hidden_size];

    // Call the function to update the cell state
    update_cell_state(f_t, i_t, c_prev, c_hat_t, c_t, hidden_size);

    // Print the updated cell state
    printf("Updated cell state:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("C_t[%d] = %.5f\n", i, c_t[i]);
    }

    return 0;
}
