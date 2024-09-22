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

int main() {
    // Define a 2x3 matrix
    float matrix[] = {1, 2, 3, 4, 5, 6};
    
    // Define a vector of size 3
    float vector[] = {1, 2, 3};
    
    // Result will be a vector of size 2
    float result[2];

    // Perform matrix-vector multiplication
    matvec_mul(matrix, vector, result, 2, 3);

    // Print the result
    printf("Matrix-Vector Multiplication Result:\n");
    for (int i = 0; i < 2; i++) {
        printf("%.2f\n", result[i]);
    }

    return 0;
}
