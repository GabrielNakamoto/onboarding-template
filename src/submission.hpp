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
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), buffer(rows*cols, 0.0f) {
    #pragma omp parallel
    {
      // force thread initializaton/warmup
      #pragma omp barrier
    }
  }

  std::size_t get_rows() const { return rows_; }
  std::size_t get_cols() const { return cols_; }

  double& operator()(std::size_t i, std::size_t j)      { return buffer[i*cols_ + j]; }
  double operator()(std::size_t i, std::size_t j) const { return buffer[i*cols_ + j]; }
  double *data()              { return &buffer[0]; }
  const double *data() const  { return &buffer[0]; }
};

inline static __m256d kernel(__m256d mid, __m256d top, __m256d bottom, __m256d right, __m256d left) {
  static const __m256d factor_outer = _mm256_set1_pd(0.125f), factor_inner  = _mm256_set1_pd(0.5f);

  // individual muls instead of fmad to avoid dependency chain of acc
  mid     = _mm256_mul_pd(mid, factor_inner);
  top     = _mm256_mul_pd(top, factor_outer);
  bottom  = _mm256_mul_pd(bottom, factor_outer);
  left    = _mm256_mul_pd(left, factor_outer);
  right   = _mm256_mul_pd(right, factor_outer);

  return _mm256_add_pd(top, _mm256_add_pd(bottom, _mm256_add_pd(left, _mm256_add_pd(right, mid))));
}

void apply_stencil(const Grid& __restrict__ old_grid, Grid& __restrict__ new_grid) {
  std::size_t rows = old_grid.get_rows(), cols = old_grid.get_cols();

  const double *__restrict__ cells = old_grid.data();
  double *__restrict__ dst = new_grid.data();

  memcpy(dst, cells, cols * sizeof(double));
  memcpy(dst+(rows-1)*cols, cells+(rows-1)*cols, cols * sizeof(double));

  new_grid(0,0)       = old_grid(0,0);
  new_grid(rows-1,cols-1)  = old_grid(rows-1,cols-1);

  if (cols < 3 || rows < 3) return;

  #pragma omp parallel for schedule(static)
  for (std::size_t row=1; row<rows-1; ++row) {
    new_grid(row,0)       = old_grid(row,0);
    new_grid(row,cols-1)  = old_grid(row,cols-1);

    for (std::size_t col=1; col<cols-1; col+=4) {
      __m256d top, bottom, mid, left, right;

      // handle unaligned final batch with explicit intrinsics to avoid
      // handling outside parallelized hot loop
      if (col + 4 >= cols-1 && (cols - 2) % 4 > 0) {
        long long masks[4] = {0, 0, 0, 0};
        for (std::size_t m=0; m<(cols-2)%4; ++m) masks[m] = -1;

        // NOTE: reverse order args from memory placement for mask registers
        __m256i mask = _mm256_set_epi64x(masks[3], masks[2], masks[1], masks[0]);

        top     = _mm256_maskload_pd(cells + ((row-1)  * cols) + col, mask);
        bottom  = _mm256_maskload_pd(cells + ((row+1)  * cols) + col, mask);
        mid     = _mm256_maskload_pd(cells + (row      * cols) + col, mask);
        left    = _mm256_maskload_pd(cells + (row      * cols) + col - 1, mask);
        right   = _mm256_maskload_pd(cells + (row      * cols) + col + 1, mask);

        __m256d out = kernel(mid, top, bottom, right, left);
        _mm256_maskstore_pd(dst + row*cols + col, mask, out);
      } else {
        top     = _mm256_loadu_pd(cells + ((row-1)  * cols) + col);
        bottom  = _mm256_loadu_pd(cells + ((row+1)  * cols) + col);
        mid     = _mm256_loadu_pd(cells + (row      * cols) + col);
        left    = _mm256_loadu_pd(cells + (row      * cols) + col - 1);
        right   = _mm256_loadu_pd(cells + (row      * cols) + col + 1);

        __m256d out = kernel(mid, top, bottom, right, left);
        _mm256_storeu_pd(dst + row*cols + col, out);
      }
    }
  }
}
