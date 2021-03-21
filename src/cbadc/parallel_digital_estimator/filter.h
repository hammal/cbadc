#ifndef FILTER
#define FILTER

namespace CBC
{
    class Filter
    {
    public:
        Filter(int K1, int K2, int M, int N, int L);
        ~Filter();
        virtual void compute_batch();
        void new_batch();
        void input(char *s);
        double *output();
        int number_of_controls();
        int number_of_states();
        int number_of_inputs();
        int batch_size();
        int look_ahead();
        int size();

    protected:
        int batch_start_position = 0;
        int number_of_control_signals = 0;
        int K1, K2, K3, M, N, L;
        char *control_signal;
        double *estimate;
    };
} // namespace CBC
#endif