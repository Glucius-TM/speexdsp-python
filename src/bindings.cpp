#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

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
          frame_samples_(static_cast<std::size_t>(frame_size) * static_cast<std::size_t>(mics)),
          output_(frame_samples_),
          impl_(EchoCanceller::create(frame_size, filter_length, sample_rate, mics, speakers)) {}

    static std::unique_ptr<PyEchoCanceller> create(int frame_size = 256,
                                                    int filter_length = 2048,
                                                    int sample_rate = 16000,
                                                    int mics = 1,
                                                    int speakers = 1) {
        return std::make_unique<PyEchoCanceller>(frame_size, filter_length, sample_rate, mics, speakers);
    }

    py::array process(py::buffer near_buf, py::buffer far_buf) {
        ensure_alive();

        auto near = near_buf.request();
        auto far = far_buf.request();
        validate_input_buffer(near, "near");
        validate_input_buffer(far, "far");

        {
            py::gil_scoped_release release;
            process_raw(static_cast<const int16_t*>(near.ptr), static_cast<const int16_t*>(far.ptr), output_.data());
        }

        return py::array(
            py::dtype::of<int16_t>(),
            {static_cast<py::ssize_t>(frame_samples_)},
            {static_cast<py::ssize_t>(kSampleBytes)},
            output_.data(),
            py::cast(this)
        );
    }

    void process_into(py::buffer near_buf, py::buffer far_buf, py::buffer out_buf) {
        ensure_alive();

        auto near = near_buf.request();
        auto far = far_buf.request();
        auto out = out_buf.request();
        validate_input_buffer(near, "near");
        validate_input_buffer(far, "far");
        validate_output_buffer(out, "out");

        {
            py::gil_scoped_release release;
            process_raw(static_cast<const int16_t*>(near.ptr), static_cast<const int16_t*>(far.ptr), static_cast<int16_t*>(out.ptr));
        }
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
    void validate_buffer_layout(const py::buffer_info& buf, const char* name, bool writable = false) const {
        if (buf.ndim != 1) {
            throw py::type_error(std::string("expected one-dimensional contiguous int16 ") + name + " buffer");
        }
        if (buf.itemsize != static_cast<py::ssize_t>(kSampleBytes)) {
            throw py::type_error(std::string("expected int16 ") + name + " buffer");
        }
        if (!buf.format.empty() && buf.format != py::format_descriptor<int16_t>::format()) {
            throw py::type_error(std::string("expected int16 ") + name + " buffer");
        }
        if (static_cast<std::size_t>(buf.size) != frame_samples_) {
            throw py::type_error(std::string("expected frame_size * mics int16 samples in ") + name + " buffer");
        }
        if (!buf.strides.empty() && static_cast<std::size_t>(buf.strides[0]) != kSampleBytes) {
            throw py::type_error(std::string("expected C-contiguous int16 ") + name + " buffer");
        }
        (void)writable;
    }

    void validate_input_buffer(const py::buffer_info& buf, const char* name) const {
        validate_buffer_layout(buf, name, false);
    }

    void validate_output_buffer(const py::buffer_info& buf, const char* name) const {
        validate_buffer_layout(buf, name, true);
    }

    void process_raw(const int16_t* near, const int16_t* far, int16_t* out) const {
        impl_->process(near, far, out);
    }

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
    std::size_t frame_samples_;
    std::vector<int16_t> output_;
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
             py::arg("near").noconvert(), py::arg("far").noconvert(),
             R"pbdoc(Cancel echo using contiguous int16 buffers. The returned array is a zero-copy view over an internal reusable buffer.)pbdoc")
        .def("process_into", &PyEchoCanceller::process_into,
             py::arg("near").noconvert(), py::arg("far").noconvert(), py::arg("out").noconvert(),
             R"pbdoc(Cancel echo in-place into a caller-provided contiguous int16 buffer.)pbdoc")
        .def("reset", &PyEchoCanceller::reset)
        .def("destroy", &PyEchoCanceller::destroy)
        .def_property_readonly("ok", &PyEchoCanceller::ok)
        .def_property_readonly("frame_size", &PyEchoCanceller::frame_size)
        .def_property_readonly("filter_length", &PyEchoCanceller::filter_length)
        .def_property_readonly("sample_rate", &PyEchoCanceller::sample_rate)
        .def_property_readonly("mics", &PyEchoCanceller::mics)
        .def_property_readonly("speakers", &PyEchoCanceller::speakers);
}
