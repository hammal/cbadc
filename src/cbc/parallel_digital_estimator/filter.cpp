#include "filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace CBC;

Filter::Filter(int K1, int K2, int M, int N, int L) : K1(K1), K2(K2), K3(K1 + K2), M(M), N(N), L(L)
{
    // Initialize control signal buffer
    control_signal = (char *)std::malloc(M * (K3 + 1) * sizeof(char));
    if (control_signal == NULL)
    {
        std::cout << "Control signal buffer failed to allocate." << std::endl;
        exit(-1);
    }
    estimate = (double *)std::malloc(L * K1 * sizeof(double));
    if (estimate == NULL)
    {
        std::cout << "Estimates buffer failed to allocate."
                  << std::endl;
        exit(-1);
    }
}

Filter::~Filter()
{
    free(control_signal);
    free(estimate);
}

void Filter::new_batch()
{
    batch_start_position = (batch_start_position + K1) % K3;
    number_of_control_signals -= K1;
    for (int k = 0; k < K1; k++)
    {
        for (int l = 0; l < L; l++)
        {
            estimate[k * L + l] = 0;
        }
    }
}

void Filter::input(char *s)
{
    int offset = (M * (batch_start_position + number_of_control_signals)) % (M * (K3 + 1));
    number_of_control_signals++;
    for (int i = 0; i < M; i++)
    {
        control_signal[offset + i] = s[i];
    }
}

double *Filter::output()
{
    // int offset = (batch_start_position * L) % (L * K1);
    return estimate; //+ offset;
}

int Filter::number_of_controls()
{
    return M;
}

int Filter::number_of_states()
{
    return N;
}

int Filter::number_of_inputs()
{
    return L;
}

int Filter::batch_size()
{
    return K1;
}

int Filter::look_ahead()
{
    return K2;
}

int Filter::size()
{
    return K3;
}
