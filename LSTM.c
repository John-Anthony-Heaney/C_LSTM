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




void generate_cosine_wave(float* series, int length, float amplitude, float frequency) {
    for (int t = 0; t < length; t++) {
        series[t] = amplitude * cos(2 * M_PI * frequency * t / length);
    }
}

void generate_logarithmic_series(float* series, int length, float base) {
    for (int t = 1; t < length; t++) {
        series[t] = log(t) / log(base);
    }
    series[0] = 0.0; // Log(0) is undefined, set to 0
}

void generate_quadratic_series(float* series, int length, float a, float b, float c) {
    for (int t = 0; t < length; t++) {
        series[t] = a * t * t + b * t + c;
    }
}

// Function to generate an exponential decay curve
void generate_exponential_decay(float* series, int length, float A, float b) {
    for (int t = 0; t < length; t++) {
        series[t] = A * exp(-b * t);
    }
}


// Large random oscillations function
void generate_large_random_oscillations(float* series, int length, float amplitude, float frequency, float noise_factor) {
    for (int t = 0; t < length; t++) {
        float sine_value = amplitude * sin(2 * M_PI * frequency * t / length);
        float random_noise = ((rand() % 200) / 100.0 - 1.0) * noise_factor; // Random noise between -noise_factor and +noise_factor
        series[t] = sine_value + random_noise;
    }
}


void generate_gamma_like_curve(float* series, int length, float a, float b) {
    for (int t = 0; t < length; t++) {
        series[t] = pow(t, a) * exp(-b * t);
    }
}


// Function to write the curve data to a file
void write_curve_to_file(const char* filename, float* series, int length) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file for writing\n");
        return;
    }

    // Write the data in columns: Time and Value
    for (int t = 0; t < length; t++) {
        fprintf(file, "%d %.5f\n", t, series[t]);
    }

    fclose(file);
    printf("Data has been written to %s\n", filename);
}

// Function to plot the curve using gnuplot
void plot_with_gnuplot(const char* filename) {
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");

    // Set plot title and labels
    fprintf(gnuplotPipe, "set title 'Exponential Decay Curve'\n");
    fprintf(gnuplotPipe, "set xlabel 'Time'\n");
    fprintf(gnuplotPipe, "set ylabel 'Value'\n");
    fprintf(gnuplotPipe, "set grid\n");

    // Plot the data from the file
    fprintf(gnuplotPipe, "plot '%s' using 1:2 title 'Exponential Decay' with lines\n", filename);

    fflush(gnuplotPipe); // Ensure the commands are sent to gnuplot
    pclose(gnuplotPipe); // Close the pipe when done
}




int main() {
    int length = 100;
    float series[length];

    // Parameters for the exponential decay
    float A = 10.0;  // Initial amplitude
    float b = 0.1;   // Decay rate

    // Generate the exponential decay curve
    generate_exponential_decay(series, length, A, b);

    // Write the data to a file
    const char* filename = "exponential_decay_curve.dat";
    write_curve_to_file(filename, series, length);

    // Plot the curve using gnuplot
    plot_with_gnuplot(filename);

    return 0;
}