// Kernel A sum by row
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16

#define TM 8
#define TN SG_SZ
#define TK 32

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T, size_t M, size_t K>
void sum_rows_ref(
    accessor<T, 2, access::mode::read, access::target::host_buffer> A,
    accessor<int, 1, access::mode::read, access::target::host_buffer>
        sum_rows) {
  int sum_rows_ref[M] = {0};
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      sum_rows_ref[i] += A[i][j];
    }
    auto diff = sum_rows[i] - sum_rows_ref[i];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
  }
}

template <typename T, size_t M, size_t K>
void matrix_sum_rows(queue q, big_matrix<T, M, K> &A, nd_range<2> &r) {
  buffer<int8_t, 2> bufA(A.get_data(), range<2>(M, K));
  // size of vector is known because SG size of set by the user in this case
  int sum_rows[M] = {0};
  buffer<int> sum_rows_v(sum_rows, M); // there are total of M rows
  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);

     auto v = sum_rows_v.get_access<access::mode::atomic>(cgh);

     cgh.parallel_for<class add_matrix>(
         r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();

           // TM = 8, TK = 32
           joint_matrix<T, TM, TK, matrix_layout::row_major> sub_a(sg);

           joint_matrix_load(sg, sub_a,
                             accA.get_pointer() + (global_idx * TM * K) +
                                 TK,
                             K, matrix_layout::row_major);
           // calculate sum of rows in sum_rows_v[8], there are 8 rows in sub_a
           int32_t sum_local_rows[M] = {0}; // 8 local rows, M total
           // sub_a has 8x32 elements, 16 elements per WI, 2 per WI per row
           auto data = sub_a.get_wi_data();

           size_t
               global_index; // Index into the result array that holds the sums.
           
           // clang-format off
           /* x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              <---------------------------------  SG1 --------------------------------->

              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
              <0> <1>  <2> <3>    <4> <5>  <6> <7>  ..... WORK ITEMS

              Each work item has 16 elements <8 rows and 2 cols of the original matrix>
              the data_slice in holds the matrix elements in the following order:

              0 0  0 0
                /
               /
              1 1  1 1
                /
               /
              2 2  2 2
                /
               / 
              3 3  3 3 
              
              W0 --> 0 0 1 1 2 2 3 3 .... 7 7
            */ 
            // clang-format on

           //  each WI calculates local sum of rows
           // TM =8, TK = 32
           for (int row = 0; row < TM * (TK / (SG_SZ * 2));
                row++) { // there are 8 rows
             for (int i = 0; i < data.length() / (SG_SZ / 2);
                  i++) { // 2 per row
               // i*SG_SIZE index is found based on the round robin
               // distribution we are using in the implementation
               global_index = row + global_idx * TM;
               const auto data_index = row * 2 + i;
               sum_local_rows[global_index] += data[data_index];
               /* WI_global0 --> row 0 --> i [0,1] data [0, 1] --> local 0
                  global 0 row 1 --> i [0,1] data [2, 3] --> local 1  global 1
                          row 2 --> data [4, 5] --> local 2 global 2
                          ..
                          row 7 --> data [14, 15] --> local 7 global 7

                  WI_global1 --> row 0 --> data [0, 1] --> local 8 global 8
                          row 1 --> data [2, 3] --> local 9  global 9
                          row 2 --> data [4, 5] --> local 10 global 10
                          ..
                          row 7 --> data [14, 15] --> local 15 global 15

               */
             }
             sum_local_rows[global_index] = reduce_over_group(
                 sg, sum_local_rows[global_index], sycl::plus<>());

             // only Groups leader perform the global reduction
             if (global_idy % SG_SZ == 0) {
               atomic_fetch_add(v[global_index], sum_local_rows[global_index]);
             }
           }
         }); // parallel for
   }).wait();
  sum_rows_ref<T, M, K>(bufA.get_access<access::mode::read>(),
                        sum_rows_v.get_access<access::mode::read>());
}


static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_K = TK * 2;
int8_t A[MATRIX_M][MATRIX_K];

int main() {
  big_matrix<int8_t, MATRIX_M, MATRIX_K> MA((int8_t *)&A);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeK = MATRIX_K / TK;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeK * SG_SZ}, {1, 1 * SG_SZ});

  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = i;
    }
  }

  matrix_sum_rows<int8_t, MATRIX_M, MATRIX_K>(q, MA, r);

  std::cout << "Passed\n";

  return 0;
}
