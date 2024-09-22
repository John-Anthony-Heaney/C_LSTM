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

int main() {
    // Test values
    float test_values[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    int num_values = sizeof(test_values) / sizeof(test_values[0]);

    // Test sigmoid function
    printf("Testing Sigmoid Function:\n");
    for (int i = 0; i < num_values; i++) {
        float x = test_values[i];
        printf("sigmoid(%.2f) = %.5f\n", x, sigmoid(x));
    }

    // Test tanh function
    printf("\nTesting Tanh Function:\n");
    for (int i = 0; i < num_values; i++) {
        float x = test_values[i];
        printf("tanh(%.2f) = %.5f\n", x, tanh_activation(x));
    }

    return 0;
}
