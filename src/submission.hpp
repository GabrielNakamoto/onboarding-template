#pragma once
#include <immintrin.h>
#include <cstddef>
#include <vector>
#include <cstring>

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> buffer;

public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), buffer(rows*cols,0.0f) {}

  std::size_t get_rows() const { return rows_; }
  std::size_t get_cols() const { return cols_; }

  double& operator()(std::size_t i, std::size_t j)      { return buffer[i*cols_ + j]; }
  double operator()(std::size_t i, std::size_t j) const { return buffer[i*cols_ + j]; }
  double *data()              { return &buffer[0]; }
  const double *data() const  { return &buffer[0]; }
};

void apply_stencil(const Grid& __restrict__ old_grid, Grid& __restrict__ new_grid) {
  std::size_t rows = old_grid.get_rows(), cols = old_grid.get_cols();

  const double *cells = old_grid.data();
  double *dst = new_grid.data();

  memcpy(dst, cells, cols * sizeof(double));
  memcpy(dst+(rows-1)*cols, cells+(rows-1)*cols, cols * sizeof(double));

  #pragma omp parallel for
  for (std::size_t i=0; i<rows; ++i) {
    new_grid(i,0)       = old_grid(i,0);
    new_grid(i,cols-1)  = old_grid(i,cols-1);
  }

  if (cols < 3 || rows < 3) return;

  std::size_t aligned_cols = (cols - 2) - ((cols - 2) % 4);
  const __m256d factor_outer  = _mm256_set1_pd(0.125f);
  const __m256d factor_inner  = _mm256_set1_pd(0.5f);

  #pragma omp parallel for
  for (std::size_t row=1; row<rows-1; ++row) {
    for (std::size_t col=1; col<aligned_cols; col+=4) {
      // vectorize patches of 4 (64x4=256 ymm) inner fmas
      __m256d top     = _mm256_loadu_pd(cells + ((row-1)  * cols) + col);
      __m256d bottom  = _mm256_loadu_pd(cells + ((row+1)  * cols) + col);
      __m256d mid     = _mm256_loadu_pd(cells + (row      * cols) + col);
      __m256d left    = _mm256_loadu_pd(cells + (row      * cols) + col - 1);
      __m256d right   = _mm256_loadu_pd(cells + (row      * cols) + col + 1);

      mid = _mm256_mul_pd(mid, factor_inner);
      mid = _mm256_fmadd_pd(top,    factor_outer, mid);
      mid = _mm256_fmadd_pd(bottom, factor_outer, mid);
      mid = _mm256_fmadd_pd(left,   factor_outer, mid);
      mid = _mm256_fmadd_pd(right,  factor_outer, mid);

      _mm256_storeu_pd(dst + row*cols + col, mid);
    }
  }

  // handle misalignment
  if ((cols - 2) % 4 > 0) {
    for (std::size_t row=1; row<rows-1; ++row) {
      for (std::size_t col=std::max(aligned_cols, (std::size_t)1); col < cols-1; ++col) {
        dst[row*cols + col] = (0.5 * cells[row*cols + col]) + 
                      0.125 * (cells[(row-1)*cols + col] + cells[(row+1)*cols + col]
                              +cells[row*cols + (col-1)] + cells[row*cols + (col+1)]);
      }
    }
  }
}
