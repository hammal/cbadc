#include "digital_estimator.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define control_signal_index(k) (((k + batch_start_position)) % K3 + 1)

using namespace CBC;

DigitalEstimator::DigitalEstimator(double *Af, double *Ab, double *Bf, double *Bb, double *W, int K1, int K2, int M, int N, int L) : Filter(K1, K2, M, N, L), Af{Af}, Ab{Ab}, Bf{Bf}, Bb{Bb}, W{W}
{
    mean_vector = (double *)std::malloc(N * (K1 + 1) * sizeof(double));
    if (mean_vector == NULL)
    {
        std::cout << "Mean buffer failed to allocate." << std::endl;
        exit(-1);
    }
}

DigitalEstimator::~DigitalEstimator()
{
    free(mean_vector);
}

void DigitalEstimator::compute_batch()
{
    // compute lookahead
    int end_index = (K1 + 1) * N;
    int n, nn, m, l, k1, k2, k3;
    for (k1 = K3; k1 > K1; k1--)
    {
        for (n = 0; n < N; n++)
        {
            int index = end_index + n - N;
            int index_n = n * N;
            // the index - 1 is because we don't want to overwrite
            mean_vector[index - 1] = 0;
            for (m = 0; m < M; m++)
            {
                mean_vector[index] = Bb[index_n + m] * control_signal[control_signal_index(k1) * M + m];
            }
            for (nn = 0; nn < N; nn++)
            {
                mean_vector[index] = Ab[index_n + n] * mean_vector[index + 2 * N + nn];
            }
        }
        for (n = 0; n < N; n++)
        {
            int index = end_index + n;
            mean_vector[index] = mean_vector[index - N];
        }
    }
}