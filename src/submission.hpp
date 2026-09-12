#pragma once
#include <immintrin.h>
#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
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

  for (std::size_t i=0; i<rows; ++i) {
    new_grid(i,0) = old_grid(i,0);
    new_grid(i,cols-1) = old_grid(i,cols-1);
  }

  for (std::size_t row=1; row<rows-1; ++row) {
    for (std::size_t col=1; col<cols-1; col+=4) {
      // start by just vectorizing top/bottom
      __m256d top     = _mm256_load_pd(cells + (row-1)  * cols + col);
      __m256d bottom  = _mm256_load_pd(cells + (row+1)  * cols + col);
      __m256d mid     = _mm256_load_pd(cells + row      * cols + col);
      __m256d factor_outer  = _mm256_set1_pd(0.126);
      __m256d factor_inner  = _mm256_set1_pd(0.5);

      mid = _mm256_mul_pd(mid, factor_inner);
      mid = _mm256_fmadd_pd(top,    factor_outer, mid);
      mid = _mm256_fmadd_pd(bottom, factor_outer, mid);

      alignas(32) double scalars[4];
      _mm256_store_pd(scalars, mid);


      const double *tile_row = cells + row*cols + col-1;
      for (std::size_t k=1; k<5; ++k)
        scalars[k] += (tile_row[k-1] + tile_row[k+1]) * 0.125f;

      memcpy(dst + row*cols + col, scalars, 4 * sizeof(double));
    }
  }
}
