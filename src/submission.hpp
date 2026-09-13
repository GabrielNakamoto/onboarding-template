#pragma once
#include <immintrin.h>
#include <cstddef>
#include <vector>
#include <cstring>

// To try:
// - aligned buffer allocation to reduce intrinsics overhead
// - different iteration/loop orders
// - tiling to improve cache locality


class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> buffer;

public:
  static constexpr std::size_t SIMD_STRIDE = 4;

  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), buffer(rows*cols, 0.0f) {}

  std::size_t get_rows() const { return rows_; }
  std::size_t get_cols() const { return cols_; }

  double& operator()(std::size_t i, std::size_t j)      { return buffer[i*cols_ + j]; }
  double operator()(std::size_t i, std::size_t j) const { return buffer[i*cols_ + j]; }
  double *data()              { return &buffer[0]; }
  const double *data() const  { return &buffer[0]; }
};

void apply_stencil(const Grid& __restrict__ old_grid, Grid& __restrict__ new_grid) {
  std::size_t rows = old_grid.get_rows(), cols = old_grid.get_cols();

  const double *__restrict__ cells = old_grid.data();
  double *__restrict__ dst = new_grid.data();

  #pragma omp parallel for schedule(static)
  for (std::size_t row=0; row<rows; ++row) {
    for (std::size_t col=0; col<cols; col+=Grid::SIMD_STRIDE) {
      bool tail = col + Grid::SIMD_STRIDE > cols - 1;
      if (tail) col = cols - Grid::SIMD_STRIDE - 1;

      __m256d top, bottom, mid, left, right;

      top     = _mm256_loadu_pd(cells + ((row-1)  * cols) + col);
      bottom  = _mm256_loadu_pd(cells + ((row+1)  * cols) + col);
      mid     = _mm256_loadu_pd(cells + (row      * cols) + col);
      left    = _mm256_loadu_pd(cells + (row      * cols) + col - 1);
      right   = _mm256_loadu_pd(cells + (row      * cols) + col + 1);

      mid     = _mm256_mul_pd(mid, _mm256_set1_pd(0.5f));
      top     = _mm256_mul_pd(top, _mm256_set1_pd(0.125f));
      bottom  = _mm256_mul_pd(bottom, _mm256_set1_pd(0.125f));
      left    = _mm256_mul_pd(left, _mm256_set1_pd(0.125f));
      right   = _mm256_mul_pd(right, _mm256_set1_pd(0.125f));
  
      __m256d out = _mm256_add_pd(top, _mm256_add_pd(bottom, _mm256_add_pd(left, _mm256_add_pd(right, mid))));
      _mm256_storeu_pd(dst + row*cols + col, out);

      if (tail) break;
    }
  }

  memcpy(dst, cells, cols * sizeof(double));
  memcpy(dst+(rows-1)*cols, cells+(rows-1)*cols, cols * sizeof(double));

  for (std::size_t i=0; i<rows; ++i) {
    new_grid(i,0)       = old_grid(i,0);
    new_grid(i,cols-1)  = old_grid(i,cols-1);
  }
}
