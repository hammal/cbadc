#include "parallel_digital_estimator.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define control_signal_index(kTemp) (((kTemp + batch_start_position)) % (K3 + 1))
using namespace CBC;

ParallelDigitalEstimator::ParallelDigitalEstimator(complex *lambda_f, complex *lambda_b, complex *ff, complex *fb, complex *W, int K1, int K2, int M, int N, int L) : Filter::Filter(K1, K2, M, N, L), lambda_f{lambda_f}, lambda_b{lambda_b}, ff{ff}, fb{fb}, W{W}
{
    inital_mean = (complex *)std::malloc(N * sizeof(complex));
    if (inital_mean == NULL)
    {
        std::cout << "intial mean failed to allocate." << std::endl;
        exit(-1);
    }
}

ParallelDigitalEstimator::~ParallelDigitalEstimator()
{
    free(inital_mean);
}

void ParallelDigitalEstimator::compute_batch()
{
    std::cout << "Computing Batch: " << std::endl;
#pragma omp parallel for
    for (int n = 0; n < 2 * N; n++)
    {

        if (n < N)
        {
            // Forward recursion
            complex mean = inital_mean[n];
            for (int k = 1; k < K1 + 1; k++)
            {
                mean = lambda_f[n] * mean;
                for (int m = 0; m < M; m++)
                {
                    if (control_signal[control_signal_index(k - 1) * M + m])
                    {
                        mean += ff[n * M + m];
                    }
                    else
                    {
                        mean -= ff[n * M + m];
                    }
                }

                for (int l = 0; l < L; l++)
                {
                    double temp = (W[n * L + l] * mean).real();
#pragma omp critical
                    {
                        estimate[(k - 1) * L + l] += temp;
                    }
                }
            }
            inital_mean[n] = mean;
        }
        else
        {
            // Backward recursion
            complex mean = 0;
            n -= N;
            for (int k = K3; k > 1; k--)
            {
                mean = lambda_b[n] * mean;
                for (int m = 0; m < M; m++)
                {
                    if (control_signal[control_signal_index(k) * M + m])
                    {
                        mean += fb[n * M + m];
                    }
                    else
                    {
                        mean -= fb[n * M + m];
                    }
                }
                if (k < K1)
                {
                    for (int l = 0; l < L; l++)
                    {
                        double temp = (W[n * L + l] * mean).real();
#pragma omp critical
                        {
                            estimate[(k - 1) * L + l] += temp;
                        }
                    }
                }
            }
        }
    }
}