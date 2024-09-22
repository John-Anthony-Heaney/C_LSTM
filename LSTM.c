#include <stdio.h>
#include <stdlib.h>
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



// Functions to generate series
void generate_sine_wave(float* series, int length, float amplitude, float frequency) {
    for (int t = 0; t < length; t++) {
        series[t] = amplitude * sin(2 * M_PI * frequency * t / length);
    }
}

void generate_linear_series(float* series, int length, float slope, float intercept) {
    for (int t = 0; t < length; t++) {
        series[t] = slope * t + intercept + ((rand() % 100) / 100.0 - 0.5);
    }
}

void generate_random_walk(float* series, int length) {
    series[0] = 0.0;
    for (int t = 1; t < length; t++) {
        series[t] = series[t - 1] + ((rand() % 100) / 100.0 - 0.5);
    }
}

void generate_random_series(float* series, int length) {
    for (int t = 0; t < length; t++) {
        series[t] = (rand() % 100) / 100.0;
    }
}

// Write the data to a file
void write_time_series_to_file(const char* filename, float* series1, float* series2, float* series3, float* series4, int length) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file for writing\n");
        return;
    }

    // Write the data in columns: Time, Series1, Series2, Series3, Series4
    for (int t = 0; t < length; t++) {
        fprintf(file, "%d %.5f %.5f %.5f %.5f\n", t, series1[t], series2[t], series3[t], series4[t]);
    }

    fclose(file);
    printf("Data has been written to %s\n", filename);
}


// Function to plot the data using gnuplot
void plot_with_gnuplot() {
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");

    // Set plot title and labels
    fprintf(gnuplotPipe, "set title 'Multivariate Time Series Data'\n");
    fprintf(gnuplotPipe, "set xlabel 'Time'\n");
    fprintf(gnuplotPipe, "set ylabel 'Value'\n");
    fprintf(gnuplotPipe, "set grid\n");

    // Plot the data from the file
    fprintf(gnuplotPipe, "plot 'time_series_data.dat' using 1:2 title 'Sine Wave' with lines, "
                         "'time_series_data.dat' using 1:3 title 'Linear + Noise' with lines, "
                         "'time_series_data.dat' using 1:4 title 'Random Walk' with lines, "
                         "'time_series_data.dat' using 1:5 title 'Random Series' with lines\n");

    fflush(gnuplotPipe); // Ensure the commands are sent to gnuplot
    pclose(gnuplotPipe); // Close the pipe when done
}


int main() {
    int length = 100;

    // Generate and save time series data (same as before)
    float sine_wave[length], linear_series[length], random_walk[length], random_series[length];
    generate_sine_wave(sine_wave, length, 1.0, 0.1);
    generate_linear_series(linear_series, length, 0.5, 0.0);
    generate_random_walk(random_walk, length);
    generate_random_series(random_series, length);
    write_time_series_to_file("time_series_data.dat", sine_wave, linear_series, random_walk, random_series, length);

    // Plot the data using gnuplot
    plot_with_gnuplot();

    return 0;
}