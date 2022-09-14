#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mpi4py/mpi4py.h>
#include <thallium.hpp>
#include <deque>
#define SPDLOG_FMT_EXTERNAL
#include <spdlog/spdlog.h>

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

class S4MService {

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
    tl::remote_procedure        m_broadcast_rpc;

    public:

    S4MService(py::handle mpi_comm,
               const std::string& protocol,
               int num_rpc_threads,
               bool log = false) {
        MPI_Comm comm         = MPI_COMM_NULL;
        PyObject *py_mpi_comm = mpi_comm.ptr();
        if (PyObject_TypeCheck(py_mpi_comm, &PyMPIComm_Type)) {
            comm = *PyMPIComm_Get(py_mpi_comm);
        } else {
            throw std::runtime_error(
                "S4MService should be initialized with an mpi4py communicator");
        }
        if(log) {
            spdlog::set_level(spdlog::level::trace);
        }

        MPI_Comm_size(comm, &m_size);
        MPI_Comm_rank(comm, &m_rank);

        spdlog::trace("[{}] Initializing S4MService from communicator of size {}", m_rank, m_size);

        if(num_rpc_threads == 0) num_rpc_threads = -1;
        m_engine = std::make_unique<tl::engine>(
            protocol, THALLIUM_SERVER_MODE, true, num_rpc_threads);

        m_mailbox = std::make_unique<MailBox>();

        auto self_addr = static_cast<std::string>(m_engine->self());
        spdlog::trace("[{}] S4MService address is {}", m_rank, self_addr);

        int self_addr_size = self_addr.size();
        int max_addr_size = 0;
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
        spdlog::trace("[{}] S4MService peer address lookup was successful", m_rank);

        m_broadcast_rpc = m_engine->define("s4m_broadcast",
            [this](const tl::request& req, int source, int size, const tl::bulk& remote_bulk) {
                spdlog::trace("[{}] S4MService receiving data of size {} from rank {}",
                              m_rank, size, source);
                std::vector<char> buffer(size);
                auto local_bulk = m_engine->expose({{(void*)buffer.data(), (size_t)buffer.size()}},
                                                   tl::bulk_mode::write_only);
                spdlog::trace("[{}] S4MService exposed local buffer", m_rank);
                local_bulk << remote_bulk.on(req.get_endpoint());
                spdlog::trace("[{}] S4MService successful transfer to local buffer from {}",
                              m_rank, static_cast<std::string>(req.get_endpoint()));
                m_mailbox->m_mutex.lock();
                m_mailbox->m_content.emplace_back(source, std::move(buffer));
                m_mailbox->m_mutex.unlock();
                m_mailbox->m_cv.notify_one();
                spdlog::trace("[{}] S4MService placed message from rank {} in mailbox", m_rank, source);
                req.respond();
                spdlog::trace("[{}] S4MService response successful to rank {}", m_rank, source);
            });
        spdlog::trace("[{}] S4MService broadcast RPC registration was successful", m_rank);

        MPI_Barrier(comm);
    }

    void broadcast(const py::buffer& data) const {
        std::vector<tl::async_response> responses;
        responses.reserve(m_size-1);
        py::buffer_info buf_info = data.request();
        CHECK_BUFFER_IS_CONTIGUOUS(buf_info);
        size_t size = buf_info.itemsize * buf_info.size;
        spdlog::trace("[{}] S4MService broadcast data of size {}", m_rank, size);
        void* buffer = const_cast<void*>(buf_info.ptr);
        auto bulk = m_engine->expose({{buffer, size}}, tl::bulk_mode::read_only);
        spdlog::trace("[{}] S4MService successfully exposed buffer for broadcast", m_rank);
        for(const auto& peer : m_peers) {
            responses.push_back(m_broadcast_rpc.on(peer).async(m_rank, (int)size, bulk));
        }
        spdlog::trace("[{}] S4MService successfully issued broadcast to all peers", m_rank);
        for(auto& response : responses) {
            response.wait();
        }
        spdlog::trace("[{}] S4MService successfully waited broadcast response from all peers", m_rank);
    }

    py::object receive(bool blocking) {
        std::unique_lock<tl::mutex> g(m_mailbox->m_mutex);
        if(m_mailbox->m_content.empty() && !blocking) {
            spdlog::trace("[{}] S4MService receive called: mailbox is empty", m_rank);
            return py::none();
        }
        m_mailbox->m_cv.wait(g, [&](){ return !m_mailbox->m_content.empty(); });
        const auto& message = m_mailbox->m_content.front();
        spdlog::trace("[{}] S4MService receive called: returning message from {} with size {}",
                      m_rank, message.m_source, message.m_bytes.size());
        py::tuple result = py::make_tuple(
            message.m_source,
            py::bytes(message.m_bytes.data(), message.m_bytes.size()));
        m_mailbox->m_content.pop_front();
        return result;
    }

    ~S4MService() {
        spdlog::trace("[{}] S4MService calling destructor", m_rank);
        m_mailbox.reset();
        m_peers.clear();
        m_broadcast_rpc.deregister();
        if(m_engine) {
            spdlog::trace("[{}] S4MService calling finalize", m_rank);
            m_engine->finalize();
        }
        spdlog::trace("[{}] S4MService destructor completed", m_rank);
    }

};

PYBIND11_MODULE(_s4m, s4m) {
    import_mpi4py();

    s4m.doc() = "S4M C++ extension";
    py::class_<S4MService>(s4m, "S4MService")
        .def(py::init<py::handle,const std::string&,int,bool>(),
             "comm"_a, "protocol"_a, "num_rpc_threads"_a=0, "logging"_a=false)
        .def("broadcast", &S4MService::broadcast,
             "Post data to all the other processes",
             "data"_a)
        .def("receive", &S4MService::receive,
             "Try receiving data sent by any other process",
             "blocking"_a=false);
}
