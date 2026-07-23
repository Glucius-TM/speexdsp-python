#include <cstdint>
#include <stdexcept>

#include "echo_canceller.h"

#include "speex/speex_echo.h"


class EchoCancellerImpl : public EchoCanceller
{
public:
    EchoCancellerImpl(int frame_size=256, int filter_length=2048, int sample_rate=16000, int mics=1, int speakers=1);

    ~EchoCancellerImpl();

    void process(const int16_t* near, const int16_t* far, int16_t* out);

    void reset();

private:
    void initialize_state();

    int frame_size;
    int filter_length;
    int sample_rate;
    int mics;
    int speakers;

    SpeexEchoState *st;
};


EchoCanceller* EchoCanceller::create(int frame_size, int filter_length, int sample_rate, int mics, int speakers)
{
    return new EchoCancellerImpl(frame_size, filter_length, sample_rate, mics, speakers);
}


EchoCancellerImpl::EchoCancellerImpl(int frame_size, int filter_length, int sample_rate, int mics, int speakers)
    : frame_size(frame_size),
      filter_length(filter_length),
      sample_rate(sample_rate),
      mics(mics),
      speakers(speakers),
      st(nullptr)
{
    initialize_state();
}

void EchoCancellerImpl::initialize_state()
{
    if (st != nullptr) {
        speex_echo_state_destroy(st);
        st = nullptr;
    }

    st = speex_echo_state_init_mc(frame_size, filter_length, mics, speakers);
    if (st == nullptr) {
        throw std::runtime_error("speex_echo_state_init_mc failed");
    }

    speex_echo_ctl(st, SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate);
}

EchoCancellerImpl::~EchoCancellerImpl()
{
    if (st != nullptr) {
        speex_echo_state_destroy(st);
        st = nullptr;
    }
}

void EchoCancellerImpl::reset()
{
    initialize_state();
}

void EchoCancellerImpl::process(const int16_t* near, const int16_t* far, int16_t* out)
{
    speex_echo_cancellation(st, near, far, out);
}
