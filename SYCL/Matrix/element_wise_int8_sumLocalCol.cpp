// Kernel B sum by col
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16

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

template <typename T, size_t M, size_t N>
void sum_cols_ref(
    accessor<T, 2, access::mode::read, access::target::host_buffer> B,
    accessor<int, 1, access::mode::read, access::target::host_buffer>
        sum_cols) {
  int sum_cols_ref[N] = {0};
  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < M; i++) {
      sum_cols_ref[j] += B[i][j];
    }
    auto diff = sum_cols[j] - sum_cols_ref[j];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
  }
}

template <typename T, size_t M, size_t N>
void matrix_sum_cols(queue q, big_matrix<T, M, N> &B, nd_range<2> &r) {
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(M, N));
  // size of vector is known because SG size of set by the user in this case
  int sum_cols[N] = {0};
  buffer<int> sum_cols_v(sum_cols, N); // there are total of tK/4 * 2, 16 rows
  q.submit([&](handler &cgh) {
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     auto v = sum_cols_v.get_access<access::mode::atomic>(cgh);

     cgh.parallel_for<class add_matrix>(
         r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();

           // TK = 32, TN = 16
           joint_matrix<T, TK, TN, matrix_layout::packed_b> sub_b(sg);

           joint_matrix_load(sg, sub_b,
                             accB.get_pointer() + (global_idx * (TK / 4) * N) +
                                 sg_starty / SG_SZ * TN * 4,
                             N, matrix_layout::packed_b);
            // clang-format off
           /* <    ---------------    128    ---------------------------------->
                  x x x x x x x x x x x x x x x x       ..........    x x x x x x   ^
                  x x x x x x x x x x x x x x x x       ..........    x x x x x x  16
                  x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
                  .....                                                             |
                  x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
                  x x x x x x x x x x x x x x x x       ..........    x x x x x x   v
          
                   
                    ---------------    64    ---------------->
                  x x x x   x x    ..........    x x  x x x x   ^
                  x x x x   x x    ..........    x x  x x x x   8
                  x x x x   x x    ..........    x x  x x x x   |       <-- part of (VNNI-ed) original matrix
                  .....                                         |           each SG holds
                  x x x x   x x    ..........    x x  x x x x   |
                  x x x x   x x    ..........    x x  x x x x   v
                  < WI0 >                            < WI15 >


                  <--------    16    ------------->
                  x x x     ..........    x x x   ^
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x   |       <-- part of (non-VNNI-ed) original matrix
                  .....                           |           each SG holds
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x  32
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x   |
                  x x x     ..........    x x x   v

                  If we dividie the above matrix across 16 (SG_SZ) work items,
                  each WI will hold 32 elements.  And these 32 elements will be
                  8x4 chunks as shown in the VNNI-ed matrix figure. 
           */
           // clang-format on

           int32_t sum_local_cols[N] = {0}; // 4 local cols, N total
           // sub_b has 32x16 elements, 32 elements per WI, 4 per WI per row
           auto data = sub_b.get_wi_data();

           size_t
               global_index; // Index into the result array that holds the sums.

           // each WI calculates local sum of cols
           // TK = 32
           for (int col = 0; col < data.length() / (TK / 4);
                col++) {                        // there are 4 cols
             for (int i = 0; i < TK / 4; i++) { // 8 rows per col
               // Index is found based on the round robin
               // distribution we are using in the implementation
               // clang-format off
               // In the below representation, we have WI_n_global[global_idx,global_idy]
               /*WI0_global[0, 0] --> col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 0 global 0
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 1  global 1
                                      col 2 --> data [2,6,10,...,30] --> local 2 global 2
                                      col 3 --> data [3,7,11,..,31] --> local 3 global 3
                
                WI1_global[0, 1] --> col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 4 global 4
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 5  global 5
                                      col 2 --> data [2,6,10,...,30] --> local 6 global 6
                                      col 3 --> data [3,7,11,..,31] --> local 7 global 7

                WI2_global[0, 2] -->  col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 8 global 8
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 9  global 9
                                      col 2 --> data [2,6,10,...,30] --> local 10 global 10
                                      col 3 --> data [3,7,11,..,31] --> local 11 global 11

                .....

                WI15_global[0, 15] --> col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 60 global 60
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 61  global 61
                                      col 2 --> data [2,6,10,...,30] --> local 62 global 62
                                      col 3 --> data [3,7,11,..,31] --> local 63 global 63

                
                
                WI0_global[0, 16] -->  col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 64 global 64
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 65  global 65
                                      col 2 --> data [2,6,10,...,30] --> local 66 global 66
                                      col 3 --> data [3,7,11,..,31] --> local 67 global 67
                WI1_global[0, 17] -->  col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 68 global 68
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 69  global 69
                                      col 2 --> data [2,6,10,...,30] --> local 70 global 70
                                      col 3 --> data [3,7,11,..,31] --> local 71 global 71
                ....

                WI15_global[0, 31] --> col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 124 global 124
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 125  global 125
                                      col 2 --> data [2,6,10,...,30] --> local 126 global 126
                                      col 3 --> data [3,7,11,..,31] --> local 127 global 127


                WI0_global[1, 0] -->  col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 0 global 0
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 1  global 1
                                      col 2 --> data [2,6,10,...,30] --> local 2 global 2
                                      col 3 --> data [3,7,11,..,31] --> local 3 global 3
                WI1_global[1, 1] -->  col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 4 global 4
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 5  global 5
                                      col 2 --> data [2,6,10,...,30] --> local 6 global 6
                                      col 3 --> data [3,7,11,..,31] --> local 7 global 7
                ......
                WI15_global[1, 15] -->  col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 60 global 60
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 61  global 61
                                      col 2 --> data [2,6,10,...,30] --> local 62 global 62
                                      col 3 --> data [3,7,11,..,31] --> local 63 global 63

                WI0_global[1, 16] -->  col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 64 global 64
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 65  global 65
                                      col 2 --> data [2,6,10,...,30] --> local 66 global 66
                                      col 3 --> data [3,7,11,..,31] --> local 67 global 67
                ....
                WI15_global[1, 31] --> col 0 --> i [0,1,.., 7] data [0,4,8,12,..,28] --> local 124 global 124
                                      col 1 --> i [0,1,....,7] data [1,5,9,13...,29] --> local 125  global 125
                                      col 2 --> data [2,6,10,...,30] --> local 126 global 126
                                      col 3 --> data [3,7,11,..,31] --> local 127 global 127

               */
               // clang-format on
               global_index = col + (global_idy * 4 /*VNNI_FACTOR*/);
               const auto data_index = col + (i * 4 /*VNNI factor*/);

               sum_local_cols[global_index] += data[data_index];

             } // Done Iterating over rows
               // TODO: Do we need a reduce_over_group() here for supporting
               // other row/col distributions in a different architecture?
             atomic_fetch_add(v[global_index], sum_local_cols[global_index]);
           } // iterating through columns
         }); // parallel for
   }).wait();
  sum_cols_ref<T, M, N>(bufB.get_access<access::mode::read>(),
                        sum_cols_v.get_access<access::mode::read>());
}

// TK = 32, TN = 16
static constexpr size_t MATRIX_K = TK / 4 * 2; // 16
static constexpr size_t MATRIX_N = TN * 4 * 2; // 128
int8_t B[MATRIX_K][MATRIX_N];

/* <    ---------------    128    ---------------------------------->
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   ^
   x x x x x x x x x x x x x x x x       ..........    x x x x x x  16
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
   .....                                                             |
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   v
*/
int main() {
  big_matrix<int8_t, MATRIX_K, MATRIX_N> MB((int8_t *)&B);

  size_t NDRangeK = MATRIX_K / (TK / 4);
  size_t NDRangeN = (MATRIX_N / 4) / TN;
  queue q;
  nd_range<2> r({NDRangeK, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = i;
    }
  }

  matrix_sum_cols<int8_t, MATRIX_K, MATRIX_N>(q, MB, r);

  std::cout << "Passed\n";

  return 0;
}
