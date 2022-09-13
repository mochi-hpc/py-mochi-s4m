#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mpi4py/mpi4py.h>
#include <thallium.hpp>
#include <deque>

namespace py = pybind11;
namespace tl = thallium;
using namespace pybind11::literals;
using namespace std::string_literals;

#define CHECK_BUFFER_IS_CONTIGUOUS(__buf_info__) do {        \
    ssize_t __stride__ = (__buf_info__).itemsize;            \
    for(ssize_t i=0; i < (__buf_info__).ndim; i++) {         \
        if(__stride__ != (__buf_info__).strides[i])          \
            throw std::runtime_error(                        \
                "Non-contiguous buffers not yet supported"); \
        __stride__ *= (__buf_info__).shape[i];               \
    }                                                        \
} while(0)

class S4MCommunicator {

    struct Data {

        int               m_source;
        std::vector<char> m_bytes;

        Data(int src, std::vector<char> data)
        : m_source(src)
        , m_bytes(std::move(data)) {}
        Data()                       = default;
        Data(const Data&)            = default;
        Data(Data&&)                 = default;
        Data& operator=(const Data&) = default;
        Data& operator=(Data&&)      = default;
        ~Data()                      = default;
    };

    struct MailBox {
        std::deque<Data>       m_content;
        tl::mutex              m_mutex;
        tl::condition_variable m_cv;
    };

    int                         m_rank;
    int                         m_size;
    std::unique_ptr<tl::engine> m_engine;
    std::vector<tl::endpoint>   m_peers;
    std::unique_ptr<MailBox>    m_mailbox;
    tl::remote_procedure        m_post_rpc;

    public:

    S4MCommunicator(py::handle mpi_comm,
                    const std::string& protocol,
                    int num_rpc_threads) {
        MPI_Comm comm         = MPI_COMM_NULL;
        PyObject *py_mpi_comm = mpi_comm.ptr();
        if (PyObject_TypeCheck(py_mpi_comm, &PyMPIComm_Type)) {
            comm = *PyMPIComm_Get(py_mpi_comm);
        } else {
            throw std::runtime_error(
                "S4MCommunicator should be initialized with an mpi4py communicator");
        }

        if(num_rpc_threads == 0) num_rpc_threads = -1;
        m_engine = std::make_unique<tl::engine>(
            protocol, THALLIUM_SERVER_MODE, true, num_rpc_threads);

        m_mailbox = std::make_unique<MailBox>();

        auto self_addr = static_cast<std::string>(m_engine->self());
        int self_addr_size = self_addr.size();
        int max_addr_size = 0;
        MPI_Comm_size(comm, &m_size);
        MPI_Comm_rank(comm, &m_rank);
        MPI_Allreduce(&self_addr_size, &max_addr_size, 1, MPI_INT, MPI_MAX, comm);
        std::vector<char> addr_buffer(m_size*(max_addr_size+1), 0);
        self_addr.resize(max_addr_size+1);
        MPI_Allgather(self_addr.data(), max_addr_size+1, MPI_BYTE, addr_buffer.data(),
                      max_addr_size+1, MPI_BYTE, comm);
        m_peers.reserve(m_size-1);
        for(int i=0; i < m_size; i++) {
            if(i == m_rank) continue;
            auto addr = std::string(addr_buffer.data() + i*(max_addr_size+1));
            m_peers.push_back(m_engine->lookup(addr));
        }

        m_post_rpc = m_engine->define("s4m_post",
            [this](const tl::request& req, int source, int size, const tl::bulk& remote_bulk) {
                std::vector<char> buffer(size);
                auto local_bulk = m_engine->expose({{(void*)buffer.data(), (size_t)buffer.size()}},
                                                   tl::bulk_mode::write_only);
                local_bulk << remote_bulk.on(req.get_endpoint());
                m_mailbox->m_mutex.lock();
                m_mailbox->m_content.emplace_back(source, std::move(buffer));
                m_mailbox->m_mutex.unlock();
                m_mailbox->m_cv.notify_one();
                req.respond();
            });

        MPI_Barrier(comm);
    }

    void post(const py::buffer& data) const {
        std::vector<tl::async_response> responses;
        responses.reserve(m_size-1);
        py::buffer_info buf_info = data.request();
        CHECK_BUFFER_IS_CONTIGUOUS(buf_info);
        size_t size = buf_info.itemsize * buf_info.size;
        void* buffer = const_cast<void*>(buf_info.ptr);
        auto bulk = m_engine->expose({{buffer, size}}, tl::bulk_mode::read_only);
        for(const auto& peer : m_peers) {
            responses.push_back(m_post_rpc.on(peer).async(m_rank, (int)size, bulk));
        }
        for(auto& response : responses) {
            response.wait();
        }
    }

    py::object get(bool blocking) {
        std::unique_lock<tl::mutex> g(m_mailbox->m_mutex);
        if(m_mailbox->m_content.empty() && !blocking)
            return py::none();
        m_mailbox->m_cv.wait(g, [&](){ return !m_mailbox->m_content.empty(); });
        const auto& message = m_mailbox->m_content.front();
        py::tuple result = py::make_tuple(
            message.m_source,
            py::bytes(message.m_bytes.data(), message.m_bytes.size()));
        m_mailbox->m_content.pop_front();
        return result;
    }

    ~S4MCommunicator() {
        m_mailbox.reset();
        m_peers.clear();
        m_post_rpc.deregister();
        if(m_engine)
            m_engine->finalize();
    }

};

PYBIND11_MODULE(_s4m, s4m) {
    import_mpi4py();

    s4m.doc() = "S4M C++ extension";
    py::class_<S4MCommunicator>(s4m, "S4MCommunicator")
        .def(py::init<py::handle,const std::string&,int>(),
             "comm"_a, "protocol"_a, "num_rpc_threads"_a=0)
        .def("post", &S4MCommunicator::post,
             "Post data to all the other processes",
             "data"_a)
        .def("get", &S4MCommunicator::get,
             "Get data sent by any other process",
             "blocking"_a=false);
}
