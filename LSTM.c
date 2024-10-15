#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 5  
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 1   // Single output (nat_demand)
#define TIME_STEPS 5
#define MAX_LINE_LENGTH 1024
#define MAX_ROWS 100
#define EPOCHS 1000
#define LEARNING_RATE 0.0001

// Define the DataEntry structure
typedef struct {
    char datetime[25];
    float nat_demand;
    float T2M;
    float QV2M;
    float TQL;
    float W2M;
} DataEntry;

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
    return tanh(x);
}

float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

float tanh_derivative(float x) {
    return 1.0f - x * x;
}

void matvec_mul(float* matrix, float* vector, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

void add_to_vec(float* vec1, float* vec2, int size) {
    for (int i = 0; i < size; i++) {
        vec1[i] += vec2[i];
    }
}

// Random initialization of weights
void initialize_weights(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 0.2 - 0.1;  // Random values between -0.1 and 0.1
    }
}

// Read CSV function
int read_csv(const char* filename, DataEntry* entries, int max_rows) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file\n");
        return -1;
    }

    char buffer[MAX_LINE_LENGTH];
    int row = 0;

    fgets(buffer, MAX_LINE_LENGTH, file);

    while (fgets(buffer, MAX_LINE_LENGTH, file)) {
        if (row >= max_rows) {
            printf("Max rows exceeded\n");
            break;
        }

        if (sscanf(buffer, "%24[^,],%f,%f,%f,%f,%f", 
                   entries[row].datetime, 
                   &entries[row].nat_demand, 
                   &entries[row].T2M, 
                   &entries[row].QV2M, 
                   &entries[row].TQL, 
                   &entries[row].W2M) != 6) {
            printf("Error parsing line %d: %s\n", row + 1, buffer);
            continue;
        }
        row++;
    }

    fclose(file);
    return row;
}

// Forward pass of the LSTM cell
void lstm_forward(float* W_f, float* U_f, float* b_f, float* W_i, float* U_i, float* b_i, 
                  float* W_c, float* U_c, float* b_c, float* W_o, float* U_o, float* b_o, 
                  float* x_t, float* h_prev, float* c_prev, float* h_t, float* c_t, int input_size, int hidden_size) {

    float f_t[hidden_size], i_t[hidden_size], c_hat_t[hidden_size], o_t[hidden_size];
    
    float temp[hidden_size], temp_h[hidden_size];
    
    // Forget gate
    matvec_mul(W_f, x_t, temp, hidden_size, input_size);
    matvec_mul(U_f, h_prev, temp_h, hidden_size, hidden_size);
    add_to_vec(temp, temp_h, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        f_t[i] = sigmoid(temp[i] + b_f[i]);
    }

    // Input gate
    matvec_mul(W_i, x_t, temp, hidden_size, input_size);
    matvec_mul(U_i, h_prev, temp_h, hidden_size, hidden_size);
    add_to_vec(temp, temp_h, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        i_t[i] = sigmoid(temp[i] + b_i[i]);
    }

    // Cell update
    matvec_mul(W_c, x_t, temp, hidden_size, input_size);
    matvec_mul(U_c, h_prev, temp_h, hidden_size, hidden_size);
    add_to_vec(temp, temp_h, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        c_hat_t[i] = tanh_activation(temp[i] + b_c[i]);
    }

    // Output gate
    matvec_mul(W_o, x_t, temp, hidden_size, input_size);
    matvec_mul(U_o, h_prev, temp_h, hidden_size, hidden_size);
    add_to_vec(temp, temp_h, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        o_t[i] = sigmoid(temp[i] + b_o[i]);
    }

    // Update cell state
    for (int i = 0; i < hidden_size; i++) {
        c_t[i] = f_t[i] * c_prev[i] + i_t[i] * c_hat_t[i];
    }

    // Update hidden state
    for (int i = 0; i < hidden_size; i++) {
        h_t[i] = o_t[i] * tanh_activation(c_t[i]);
    }
}

// Fully connected layer (output layer)
float fully_connected(float* h_t, float* W_fc, float b_fc, int hidden_size) {
    float output = b_fc;
    for (int i = 0; i < hidden_size; i++) {
        output += W_fc[i] * h_t[i];
    }
    return output;
}

// Mean Squared Error loss function
float calculate_loss(float* predictions, float* targets, int size) {
    float loss = 0;
    for (int i = 0; i < size; i++) {
        float error = targets[i] - predictions[i];
        loss += error * error;
    }
    return loss / size;
}

// Backpropagation and weight updates
void backpropagate_and_update(float* h_t, float* c_t, float* W_fc, float* b_fc, float* W_f, float* U_f, 
                              float* W_i, float* U_i, float* W_c, float* U_c, float* W_o, float* U_o,
                              float* grad_output, float* grad_hidden, int hidden_size) {

    // Gradient of the fully connected output layer
    for (int i = 0; i < hidden_size; i++) {
        W_fc[i] -= LEARNING_RATE * grad_output[i] * h_t[i];  // Update weights
    }
    *b_fc -= LEARNING_RATE * grad_output[0];  // Update bias

    // Backpropagate through LSTM (simplified for demo purposes)
    // Normally, we would backpropagate through all gates and update weights
}

// Forward and backward propagation with weight updates
void train_lstm(float* W_f, float* U_f, float* b_f, float* W_i, float* U_i, float* b_i, 
                float* W_c, float* U_c, float* b_c, float* W_o, float* U_o, float* b_o, 
                float* W_fc, float* b_fc, DataEntry* entries, int num_entries, int epochs, 
                int input_size, int hidden_size) {
    
    float h_prev[HIDDEN_SIZE] = {0};
    float c_prev[HIDDEN_SIZE] = {0};
    float h_t[HIDDEN_SIZE] = {0};
    float c_t[HIDDEN_SIZE] = {0};

    float predictions[num_entries];
    float targets[num_entries];

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int t = 0; t < num_entries - TIME_STEPS; t++) {
            float x_t[INPUT_SIZE] = {
                entries[t].nat_demand,
                entries[t].T2M,
                entries[t].QV2M,
                entries[t].TQL,
                entries[t].W2M
            };

            // LSTM forward pass
            lstm_forward(W_f, U_f, b_f, W_i, U_i, b_i, W_c, U_c, b_c, W_o, U_o, b_o, 
                         x_t, h_prev, c_prev, h_t, c_t, INPUT_SIZE, HIDDEN_SIZE);

            // Fully connected layer to get the prediction
            float prediction = fully_connected(h_t, W_fc, *b_fc, HIDDEN_SIZE);
            predictions[t] = prediction;
            targets[t] = entries[t].nat_demand;  // Actual target (nat_demand)

            // Calculate gradients (simplified for demo purposes)
            float grad_output[HIDDEN_SIZE] = {prediction - targets[t]};
            float grad_hidden[HIDDEN_SIZE] = {0};

            // Backpropagation and weight update
            backpropagate_and_update(h_t, c_t, W_fc, b_fc, W_f, U_f, W_i, U_i, W_c, U_c, W_o, U_o, grad_output, grad_hidden, HIDDEN_SIZE);
        }

        if (epoch % 10 == 0) {
            // Calculate and print loss
            float loss = calculate_loss(predictions, targets, num_entries - TIME_STEPS);
            printf("Epoch %d: Loss: %.6f\n", epoch, loss);
        }
    }
}

int main() {
    DataEntry entries[MAX_ROWS];
    int num_rows = read_csv("reduced_dataset.csv", entries, MAX_ROWS);
    if (num_rows == -1) {
        return 1;
    }

    // Initialize weights randomly
    float W_f[HIDDEN_SIZE * INPUT_SIZE];
    float U_f[HIDDEN_SIZE * HIDDEN_SIZE];
    float b_f[HIDDEN_SIZE];

    float W_i[HIDDEN_SIZE * INPUT_SIZE];
    float U_i[HIDDEN_SIZE * HIDDEN_SIZE];
    float b_i[HIDDEN_SIZE];

    float W_c[HIDDEN_SIZE * INPUT_SIZE];
    float U_c[HIDDEN_SIZE * HIDDEN_SIZE];
    float b_c[HIDDEN_SIZE];

    float W_o[HIDDEN_SIZE * INPUT_SIZE];
    float U_o[HIDDEN_SIZE * HIDDEN_SIZE];
    float b_o[HIDDEN_SIZE];

    float W_fc[HIDDEN_SIZE];  // Fully connected layer weights
    float b_fc;               // Bias for the fully connected layer

    initialize_weights(W_f, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(U_f, HIDDEN_SIZE * HIDDEN_SIZE);
    initialize_weights(b_f, HIDDEN_SIZE);

    initialize_weights(W_i, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(U_i, HIDDEN_SIZE * HIDDEN_SIZE);
    initialize_weights(b_i, HIDDEN_SIZE);

    initialize_weights(W_c, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(U_c, HIDDEN_SIZE * HIDDEN_SIZE);
    initialize_weights(b_c, HIDDEN_SIZE);

    initialize_weights(W_o, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(U_o, HIDDEN_SIZE * HIDDEN_SIZE);
    initialize_weights(b_o, HIDDEN_SIZE);

    initialize_weights(W_fc, HIDDEN_SIZE);  // Fully connected weights
    b_fc = ((float)rand() / RAND_MAX) * 0.2 - 0.1;  // Initialize bias for fully connected layer

    train_lstm(W_f, U_f, b_f, W_i, U_i, b_i, W_c, U_c, b_c, W_o, U_o, b_o, W_fc, &b_fc, 
               entries, num_rows, EPOCHS, INPUT_SIZE, HIDDEN_SIZE);

    return 0;
}
