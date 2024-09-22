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


int main() {
     // Example: Forget gate for one time step

    // Define the size of the input and hidden states
    int input_size = 3;   // Example: 3 input features
    int hidden_size = 2;  // Example: 2 hidden units

    // Define input x_t (current input) and h_prev (previous hidden state)
    float x_t[] = {1.0, 0.5, -1.2};  // Input vector (size 3)
    float h_prev[] = {0.1, -0.2};    // Previous hidden state (size 2)

    // Define weight matrices W_f and U_f, and bias b_f for forget gate
    float W_f[] = {0.5, 0.8, -0.3,   // Weight matrix for input (size 2x3)
                   -0.1, 0.4, 0.7};
    float U_f[] = {0.2, 0.6,         // Weight matrix for hidden state (size 2x2)
                   -0.5, 0.9};
    float b_f[] = {0.05, -0.1};      // Bias (size 2)

    // Define an array to store the forget gate output
    float f_t[hidden_size];

    // Call the forget gate function
    forget_gate(W_f, U_f, x_t, h_prev, b_f, f_t, input_size, hidden_size);

    // Print the forget gate output
    printf("Forget gate output:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("f_t[%d] = %.5f\n", i, f_t[i]);
    }

    return 0;
}
