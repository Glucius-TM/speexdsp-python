#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "echo_canceller.h"

namespace py = pybind11;

namespace {

constexpr std::size_t kSampleBytes = sizeof(int16_t);

class PyEchoCanceller {
public:
    PyEchoCanceller(int frame_size = 256,
                    int filter_length = 2048,
                    int sample_rate = 16000,
                    int mics = 1,
                    int speakers = 1)
        : frame_size_(frame_size),
          filter_length_(filter_length),
          sample_rate_(sample_rate),
          mics_(mics),
          speakers_(speakers),
          impl_(EchoCanceller::create(frame_size, filter_length, sample_rate, mics, speakers)) {}

    static std::unique_ptr<PyEchoCanceller> create(int frame_size = 256,
                                                    int filter_length = 2048,
                                                    int sample_rate = 16000,
                                                    int mics = 1,
                                                    int speakers = 1) {
        return std::make_unique<PyEchoCanceller>(frame_size, filter_length, sample_rate, mics, speakers);
    }

    py::array_t<int16_t> process(py::buffer near, py::buffer far) {
        ensure_alive();

        auto near_info = near.request();
        auto far_info = far.request();

        if (near_info.ndim != 1 || far_info.ndim != 1) {
            throw py::type_error("expected one-dimensional contiguous int16 buffers");
        }

        const std::size_t expected_samples = static_cast<std::size_t>(frame_size_) * static_cast<std::size_t>(mics_);
        const std::size_t expected_bytes = expected_samples * kSampleBytes;

        const std::size_t near_bytes = static_cast<std::size_t>(near_info.size) * static_cast<std::size_t>(near_info.itemsize);
        const std::size_t far_bytes = static_cast<std::size_t>(far_info.size) * static_cast<std::size_t>(far_info.itemsize);
        if (near_bytes != expected_bytes || far_bytes != expected_bytes) {
            throw py::type_error("expected frame_size * mics int16 samples in each input buffer");
        }

        if (near_info.itemsize != static_cast<py::ssize_t>(kSampleBytes) ||
            far_info.itemsize != static_cast<py::ssize_t>(kSampleBytes)) {
            throw py::type_error("expected numpy.ndarray[int16] inputs");
        }

        if ((!near_info.strides.empty() && static_cast<std::size_t>(near_info.strides[0]) != kSampleBytes) ||
            (!far_info.strides.empty() && static_cast<std::size_t>(far_info.strides[0]) != kSampleBytes)) {
            throw py::type_error("expected C-contiguous int16 arrays");
        }

        const auto* near_ptr = static_cast<const int16_t*>(near_info.ptr);
        const auto* far_ptr = static_cast<const int16_t*>(far_info.ptr);

        py::array_t<int16_t> out(static_cast<py::ssize_t>(expected_samples));
        auto out_info = out.request();
        auto* out_ptr = static_cast<int16_t*>(out_info.ptr);

        {
            py::gil_scoped_release release;
            impl_->process(near_ptr, far_ptr, out_ptr);
        }

        return out;
    }

    void reset() {
        ensure_alive();
        impl_->reset();
    }

    void destroy() {
        impl_.reset();
    }

    bool ok() const {
        return static_cast<bool>(impl_);
    }

    int frame_size() const { return frame_size_; }
    int filter_length() const { return filter_length_; }
    int sample_rate() const { return sample_rate_; }
    int mics() const { return mics_; }
    int speakers() const { return speakers_; }

private:
    void ensure_alive() const {
        if (!impl_) {
            throw std::runtime_error("EchoCanceller has been destroyed");
        }
    }

    int frame_size_;
    int filter_length_;
    int sample_rate_;
    int mics_;
    int speakers_;
    std::unique_ptr<EchoCanceller> impl_;
};

}  // namespace

PYBIND11_MODULE(_speexdsp, m) {
    m.doc() = "pybind11 bindings for SpeexDSP acoustic echo cancellation";

    py::class_<PyEchoCanceller>(m, "EchoCanceller")
        .def(py::init<int, int, int, int, int>(),
             py::arg("frame_size") = 256,
             py::arg("filter_length") = 2048,
             py::arg("sample_rate") = 16000,
             py::arg("mics") = 1,
             py::arg("speakers") = 1)
        .def_static("create", &PyEchoCanceller::create,
                    py::arg("frame_size") = 256,
                    py::arg("filter_length") = 2048,
                    py::arg("sample_rate") = 16000,
                    py::arg("mics") = 1,
                    py::arg("speakers") = 1)
        .def("process", &PyEchoCanceller::process,
             py::arg("near"), py::arg("far"),
             R"pbdoc(Cancel echo using near-end and far-end contiguous numpy int16 arrays.)pbdoc")
        .def("reset", &PyEchoCanceller::reset)
        .def("destroy", &PyEchoCanceller::destroy)
        .def_property_readonly("ok", &PyEchoCanceller::ok)
        .def_property_readonly("frame_size", &PyEchoCanceller::frame_size)
        .def_property_readonly("filter_length", &PyEchoCanceller::filter_length)
        .def_property_readonly("sample_rate", &PyEchoCanceller::sample_rate)
        .def_property_readonly("mics", &PyEchoCanceller::mics)
        .def_property_readonly("speakers", &PyEchoCanceller::speakers);
}
