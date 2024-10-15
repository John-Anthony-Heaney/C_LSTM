#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 5  
#define HIDDEN_SIZE 4
#define TIME_STEPS 5
#define MAX_LINE_LENGTH 1024
#define MAX_ROWS 100

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
    return tanh(x);
}

void matvec_mul(float* matrix, float* vector, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

void forget_gate(float* W_f, float* U_f, float* x_t, float* h_prev, float* b_f, float* f_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_f, x_t, temp, hidden_size, input_size);
    matvec_mul(U_f, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        f_t[i] = sigmoid(temp[i] + temp_h[i] + b_f[i]);
    }
}

void input_gate(float* W_i, float* U_i, float* x_t, float* h_prev, float* b_i, float* i_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_i, x_t, temp, hidden_size, input_size);
    matvec_mul(U_i, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        i_t[i] = sigmoid(temp[i] + temp_h[i] + b_i[i]);
    }
}

void cell_update(float* W_c, float* U_c, float* x_t, float* h_prev, float* b_c, float* c_hat_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_c, x_t, temp, hidden_size, input_size);
    matvec_mul(U_c, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        c_hat_t[i] = tanh_activation(temp[i] + temp_h[i] + b_c[i]);
    }
}

void output_gate(float* W_o, float* U_o, float* x_t, float* h_prev, float* b_o, float* o_t, int input_size, int hidden_size) {
    float temp[input_size];
    float temp_h[hidden_size];

    matvec_mul(W_o, x_t, temp, hidden_size, input_size);
    matvec_mul(U_o, h_prev, temp_h, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        o_t[i] = sigmoid(temp[i] + temp_h[i] + b_o[i]);
    }
}

void update_cell_state(float* f_t, float* i_t, float* c_prev, float* c_hat_t, float* c_t, int hidden_size) {
    for (int i = 0; i < hidden_size; i++) {
        c_t[i] = f_t[i] * c_prev[i] + i_t[i] * c_hat_t[i];
    }
}

void update_hidden_state(float* o_t, float* c_t, float* h_t, int hidden_size) {
    for (int i = 0; i < hidden_size; i++) {
        h_t[i] = o_t[i] * tanh_activation(c_t[i]);
    }
}

void lstm_cell(float* W_f, float* U_f, float* b_f,
               float* W_i, float* U_i, float* b_i,
               float* W_c, float* U_c, float* b_c,
               float* W_o, float* U_o, float* b_o,
               float* x_t, float* h_prev, float* c_prev,
               float* h_t, float* c_t,
               int input_size, int hidden_size) {

    float f_t[hidden_size], i_t[hidden_size], c_hat_t[hidden_size], o_t[hidden_size];

    forget_gate(W_f, U_f, x_t, h_prev, b_f, f_t, input_size, hidden_size);

    input_gate(W_i, U_i, x_t, h_prev, b_i, i_t, input_size, hidden_size);

    cell_update(W_c, U_c, x_t, h_prev, b_c, c_hat_t, input_size, hidden_size);

    update_cell_state(f_t, i_t, c_prev, c_hat_t, c_t, hidden_size);

    output_gate(W_o, U_o, x_t, h_prev, b_o, o_t, input_size, hidden_size);

    update_hidden_state(o_t, c_t, h_t, hidden_size);
}

typedef struct {
    char datetime[25];
    float nat_demand;
    float T2M;
    float QV2M;
    float TQL;
    float W2M;
} DataEntry;

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

int main() {
    DataEntry entries[MAX_ROWS];
    int num_rows = read_csv("reduced_dataset.csv", entries, MAX_ROWS);
    if (num_rows == -1) {
        return 1; 
    }

    float W_f[HIDDEN_SIZE * INPUT_SIZE] = {0};
    float U_f[HIDDEN_SIZE * HIDDEN_SIZE] = {0}; 
    float b_f[HIDDEN_SIZE] = {0};

    float W_i[HIDDEN_SIZE * INPUT_SIZE] = {0}; 
    float U_i[HIDDEN_SIZE * HIDDEN_SIZE] = {0}; 
    float b_i[HIDDEN_SIZE] = {0}; 

    float W_c[HIDDEN_SIZE * INPUT_SIZE] = {0}; 
    float U_c[HIDDEN_SIZE * HIDDEN_SIZE] = {0};  
    float b_c[HIDDEN_SIZE] = {0};  

    float W_o[HIDDEN_SIZE * INPUT_SIZE] = {0};  
    float U_o[HIDDEN_SIZE * HIDDEN_SIZE] = {0}; 
    float b_o[HIDDEN_SIZE] = {0};  

    float h_prev[HIDDEN_SIZE] = {0};  
    float c_prev[HIDDEN_SIZE] = {0};  
    float h_t[HIDDEN_SIZE] = {0};     
    float c_t[HIDDEN_SIZE] = {0};     

    float x_t[INPUT_SIZE] = {entries[0].nat_demand, entries[0].T2M, entries[0].QV2M, entries[0].TQL, entries[0].W2M};

    lstm_cell(W_f, U_f, b_f, W_i, U_i, b_i, W_c, U_c, b_c, W_o, U_o, b_o, x_t, h_prev, c_prev, h_t, c_t, INPUT_SIZE, HIDDEN_SIZE);

    
    printf("Hidden state after one step:\n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        printf("%f ", h_t[i]);
    }
    printf("\n");

    return 0;
}
