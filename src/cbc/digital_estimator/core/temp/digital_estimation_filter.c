#include <stdio.h>
#include <stdlib.h>
#include "digital_estimation_filter.h"

DigitalEstimationFilter create_new_filter(int M, int N, int K1, int K2, int L)
{
    int K3 = K1 + K2;
    char *ptr_to_control_signal_buffer = (char *)malloc(M * (K3 + 1) * sizeof(char));
    if (ptr_to_control_signal_buffer == NULL)
    {
        printf("Control signal buffer failed to allocate.\n");
        exit(-1);
    }
    double *ptr_to_estimates = (double *)malloc(L * K1 * sizeof(double));
    if (ptr_to_estimates == NULL)
    {
        printf("Estimates buffer failed to allocate.\n");
        exit(-1);
    }
    DigitalEstimationFilter temp;
    temp.K0 = 0;
    temp.K1 = K1;
    temp.K2 = K2;
    temp.K3 = K3;
    temp.N = N;
    temp.M = M;
    temp.L = L;
    temp.next_control_signal_index = 0;
    temp.control_signals = ptr_to_control_signal_buffer;
    temp.estimates = ptr_to_estimates;
    return temp;
}

int delete_filter(DigitalEstimationFilter *self)
{
    free(self->control_signals);
    free(self->estimates);
    return 0;
}

int append_control_signal(DigitalEstimationFilter *self, char *control_signal_sample)
{
    int index_offset = self->next_control_signal_index * self->M;
    int max = self->M * (self->K3 + 1);
    for (int i = 0; i < self->M; i++)
    {
        self->control_signals[(i + index_offset) % max] = control_signal_sample[i];
    }
    self->next_control_signal_index++;
    return 0;
}

int filter(DigitalEstimationFilter *self)
{
}
