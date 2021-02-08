#ifndef DIGITAL_ESTIMATION_FILTER
#define DIGITAL_ESTIMATION_FILTER
typedef struct digitalEstimationFilter
{
    int K0, K1, K2, K3;
    char *control_signals;
    double *estimates;
    int M, N, L;
    int next_control_signal_index;
} DigitalEstimationFilter;

DigitalEstimationFilter create_new_filter(int M, int K1, int K2, int L);

int delete_filter(DigitalEstimationFilter *self);

int append_control_signal(DigitalEstimationFilter *self, char *control_signal_sample);

#endif