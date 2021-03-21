#ifndef PARALLEL_DIGITAL_ESTIMATOR
#define PARALLEL_DIGITAL_ESTIMATOR
#include "filter.h"
#include <complex>

namespace CBC
{
    typedef std::complex<double> complex;

    class ParallelDigitalEstimator
    {
    private:
        complex *fa, *ba, *fb, *bb, *fw, *bw, *inital_mean;
        int batch_start_position = 0;
        int batch_number_of_controls = 0;
        int batch_estimates_pointer;
        int K1, K2, K3, M, N, L;
        int8_t *control_signal;
        double *estimate;
        void compute_batch();
        void reset_estimate();

    public:
        ParallelDigitalEstimator(complex *lambda_f, complex *lambda_b, complex *ff, complex *fb, complex *fw, complex *bw, int K1, int K2, int M, int N, int L);
        ~ParallelDigitalEstimator();
        void compute_new_batch();
        int number_of_controls();
        int number_of_states();
        int number_of_inputs();
        int number_of_estimates_in_batch();
        bool empty_batch();
        int number_of_control_signals();
        bool full_batch();
        int batch_size();
        int lookahead();
        int size();
        void input(int *s);
        void output(double *estimate);
    };
} // namespace CBC
#endif