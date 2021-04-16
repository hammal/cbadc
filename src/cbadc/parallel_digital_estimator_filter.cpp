#include "parallel_digital_estimator_filter.h"
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#define control_signal_index(kTemp) (((kTemp + batch_start_position)) % (K3 + 1))
using namespace CBC;

ParallelDigitalEstimator::ParallelDigitalEstimator(
    complex *forward_lambda,
    complex *backward_lambda,
    complex *forward_b,
    complex *backward_b,
    complex *forward_w,
    complex *backward_w,
    int K1, int K2, int M, int N, int L) : K1(K1), K2(K2), K3(K1 + K2), M(M), N(N), L(L)
{
  inital_mean = (complex *)std::malloc(N * sizeof(complex));
  fa = (complex *)std::malloc(N * sizeof(complex));
  ba = (complex *)std::malloc(N * sizeof(complex));
  fb = (complex *)std::malloc(M * N * sizeof(complex));
  bb = (complex *)std::malloc(M * N * sizeof(complex));
  fw = (complex *)std::malloc(N * L * sizeof(complex));
  bw = (complex *)std::malloc(N * L * sizeof(complex));
  control_signal = (int8_t *)std::malloc(M * (K3 + 1) * sizeof(int8_t));
  estimate = (double *)std::malloc(L * K1 * sizeof(double));
  if (inital_mean == NULL)
  {
    std::cout << "intial mean failed to allocate." << std::endl;
    exit(-1);
  }
  if (fa == NULL)
  {
    std::cout << "fa failed to allocate." << std::endl;
    exit(-1);
  }
  if (ba == NULL)
  {
    std::cout << "ba failed to allocate." << std::endl;
    exit(-1);
  }
  if (fb == NULL)
  {
    std::cout << "fb failed to allocate." << std::endl;
    exit(-1);
  }
  if (bb == NULL)
  {
    std::cout << "bb failed to allocate." << std::endl;
    exit(-1);
  }

  if (fw == NULL)
  {
    std::cout << "fw failed to allocate." << std::endl;
    exit(-1);
  }
  if (fb == NULL)
  {
    std::cout << "fb failed to allocate." << std::endl;
    exit(-1);
  }
  if (control_signal == NULL)
  {
    std::cout << "Control signal buffer failed to allocate." << std::endl;
    exit(-1);
  }
  if (estimate == NULL)
  {
    std::cout << "Estimates buffer failed to allocate." << std::endl;
    exit(-1);
  }
  // std::cout << "From C++" << std::endl;
  std::cout.precision(10);
  for (int n = 0; n < N; n++)
  {
    *(fa + n) = *(forward_lambda + n);
    // std::cout << "fa" << std::endl;
    // std::cout << *(fa + n) << std::endl;
    *(ba + n) = *(backward_lambda + n);
    // std::cout << "ba" << std::endl;
    // std::cout << *(ba + n) << std::endl;
    *(inital_mean + n) = 0;
    // std::cout << "inital_mean" << std::endl;
    // std::cout << *(inital_mean + n) << std::endl;
    for (int l = 0; l < L; l++)
    {
      *(fw + L * n + l) = *(forward_w + L * n + l);
      // std::cout << "fw" << std::endl;
      // std::cout << *(fw + L * n + l) << std::endl;
      *(bw + L * n + l) = *(backward_w + L * n + l);
      // std::cout << "bw" << std::endl;
      // std::cout << *(bw + L * n + l) << std::endl;
    }

    for (int m = 0; m < M; m++)
    {
      *(fb + n * M + m) = *(forward_b + n * M + m);
      // std::cout << "fb" << std::endl;
      // std::cout << *(fb + n * M + m) << std::endl;
      *(bb + n * M + m) = *(backward_b + n * M + m);
      // std::cout << "bb" << std::endl;
      // std::cout << *(bb + n * M + m) << std::endl;
    }
  }
  reset_estimate();
  batch_estimates_pointer = K1;
}

ParallelDigitalEstimator::~ParallelDigitalEstimator()
{
  free(control_signal);
  free(estimate);
  free(inital_mean);
}

void ParallelDigitalEstimator::input(int *s)
{
  int offset = (M * (batch_start_position + batch_number_of_controls)) % (M * (K3 + 1));
  for (int i = 0; i < M; i++)
  {
    *(control_signal + offset + i) = *(s + i);
    // // std::cout << *(s + i);
    // // std::cout << ", ";
  }
  // // std::cout << std::endl;
  batch_number_of_controls++;
}

void ParallelDigitalEstimator::output(double *return_array)
{
  if (batch_estimates_pointer < 0 || batch_estimates_pointer > K1 - 1)
  {
    std::cout << "batch_estimate_pointer out of bounds" << std::endl;
    throw "Batch estimator pointer out of bounds";
    // exit(-1);
  }
  for (int i = 0; i < L; i++)
  {
    *(return_array + i) = *(estimate + batch_estimates_pointer * L + i);
  }
  batch_estimates_pointer++;
}

void ParallelDigitalEstimator::reset_estimate()
{
  for (int k = 0; k < K1; k++)
  {
    for (int l = 0; l < L; l++)
    {
      estimate[k * L + l] = 0;
    }
  }
}

void ParallelDigitalEstimator::compute_new_batch()
{
  reset_estimate();
  compute_batch();
  batch_start_position = control_signal_index(K1);
  batch_number_of_controls -= K1;
  batch_estimates_pointer = 0;
}

int ParallelDigitalEstimator::number_of_estimates_in_batch()
{
  return K1 - batch_estimates_pointer;
}
bool ParallelDigitalEstimator::empty_batch()
{
  return batch_estimates_pointer > K1 - 1;
}
int ParallelDigitalEstimator::number_of_control_signals()
{
  return batch_number_of_controls;
}
bool ParallelDigitalEstimator::full_batch()
{
  return batch_number_of_controls > K3 - 1;
}

int ParallelDigitalEstimator::number_of_controls()
{
  return M;
}

int ParallelDigitalEstimator::number_of_states()
{
  return N;
}

int ParallelDigitalEstimator::number_of_inputs()
{
  return L;
}

int ParallelDigitalEstimator::batch_size()
{
  return K1;
}

int ParallelDigitalEstimator::lookahead()
{
  return K2;
}

int ParallelDigitalEstimator::size()
{
  return K3;
}

void ParallelDigitalEstimator::compute_batch()
{
  // std::cout << "Computing Batch: " << std::endl;
  // Ensure estimate is all zero
  for (int k = 0; k < K1; k++)
  {
    for (int l = 0; l < L; l++)
    {
      *(estimate + k * L + l) = 0;
    }
  }
#pragma omp parallel for
  for (int n = 0; n < 2 * N; n++)
  {

    if (n < N)
    {
      // Forward recursion
      complex mean = *(inital_mean + n);
      for (int k = 1; k < K1 + 1; k++)
      {
        mean = *(fa + n) * mean;
        for (int m = 0; m < M; m++)
        {
          if (*(control_signal + control_signal_index(k - 1) * M + m))
          {
            mean += *(fb + n * M + m);
          }
          else
          {
            mean -= *(fb + n * M + m);
          }
        }

        for (int l = 0; l < L; l++)
        {
          double temp = (*(fw + n * L + l) * mean).real();
#pragma omp critical
          {
            *(estimate + (k - 1) * L + l) += temp;
          }
        }
      }
      inital_mean[n] = mean;
    }
    else
    {
      // Backward recursion
      complex mean = 0;
      int nn = n - N;
      for (int k = K3; k > 1; k--)
      {
        mean = *(ba + nn) * mean;
        for (int m = 0; m < M; m++)
        {
          if (*(control_signal + control_signal_index(k) * M + m))
          {
            mean += *(bb + nn * M + m);
          }
          else
          {
            mean -= *(bb + nn * M + m);
          }
        }
        if (k < K1)
        {
          for (int l = 0; l < L; l++)
          {
            double temp = (*(bw + nn * L + l) * mean).real();
#pragma omp critical
            {
              *(estimate + (k - 1) * L + l) += temp;
            }
          }
        }
      }
    }
  }
}
