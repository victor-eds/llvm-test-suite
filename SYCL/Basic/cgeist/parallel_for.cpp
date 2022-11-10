// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xcgeist -gen-all-sycl-funcs %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// REQUIRES: linux
// UNSUPPORTED: hip || cuda

#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr unsigned N = 8;

void host_parallel_for(std::array<int, N> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";
  auto range = sycl::range<1>{N};

  {
    auto buf = buffer<int, 1>{A.data(), range};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for>(range, [=](sycl::id<1> id) {
        A[id] = id;
      });
    });
  }
}

int main() {
  std::array<int, N> A{0};
  host_parallel_for(A);
  for (unsigned i = 0; i < N; ++i) {
    assert(A[i] == i);
  }
  std::cout << "Test passed" << std::endl;
}
