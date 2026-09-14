#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <cstring>

class Grid {
private:
  struct FreeDeleter {
    void operator()(double* p) const noexcept { std::free(p); }
  };

  static constexpr std::size_t kAlign = 64;

  static double* alloc_zeroed(std::size_t count) {
    std::size_t bytes = (count * sizeof(double) + kAlign - 1) / kAlign * kAlign;
    if (bytes == 0) bytes = kAlign;
    void* p = std::aligned_alloc(kAlign, bytes);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, bytes);
    return static_cast<double*>(p);
  }

  std::size_t rows_;
  std::size_t cols_;
  std::unique_ptr<double[], FreeDeleter> buffer;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), buffer(alloc_zeroed(rows * cols)) {}

  std::size_t get_rows() const { return rows_; }
  std::size_t get_cols() const { return cols_; }

  double& operator()(std::size_t i, std::size_t j)       { return buffer[i*cols_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return buffer[i*cols_ + j]; }
  double       *data()       { return buffer.get(); }
  const double *data() const { return buffer.get(); }
};

static inline __m256d kernel(__m256d top, __m256d bottom, __m256d mid, __m256d left, __m256d right) {
  mid = _mm256_mul_pd(mid, _mm256_set1_pd(0.5f));
  mid = _mm256_fmadd_pd(top, _mm256_set1_pd(0.125f), mid);
  mid = _mm256_fmadd_pd(bottom, _mm256_set1_pd(0.125f), mid);
  mid = _mm256_fmadd_pd(left, _mm256_set1_pd(0.125f), mid);
  return _mm256_fmadd_pd(right, _mm256_set1_pd(0.125f), mid);
}

static inline double c_kernel(const double *in, std::size_t r, std::size_t c, std::size_t stride) {
  return 0.5f*in[r*stride+c] + 0.125f * (in[(r+1)*stride+c] + in[(r-1)*stride+c] + in[r*stride+c+1] + in[r*stride+c-1]);
}

static void stencil_impl(const std::size_t rows, const std::size_t cols, const double *__restrict__ cells, double *__restrict__ dst) {
  auto row_pair = [=](std::size_t row) {
    const double *r0 = cells + (row-1)*cols, *r1 = r0 + cols, *r2 = r1 + cols, *r3 = r2 + cols;
    double *d1 = dst + row*cols, *d2 = d1 + cols;
    auto block = [=](std::size_t col) {
      // batch kernel applications by 2x4 elements, reduces number of loads by sharing top/bottom between rows
      // and vectorize 4 cols at a time with AVX2 intrinsics
      __m256d h_top = _mm256_loadu_pd(r0 + col), h_mid = _mm256_loadu_pd(r1 + col),
              h_bottom = _mm256_loadu_pd(r2 + col), b_bottom = _mm256_loadu_pd(r3 + col);
      __m256d h_left = _mm256_loadu_pd(r1 + col - 1), h_right = _mm256_loadu_pd(r1 + col + 1);
      __m256d b_left = _mm256_loadu_pd(r2 + col - 1), b_right = _mm256_loadu_pd(r2 + col + 1);
      _mm256_storeu_pd(d1 + col, kernel(h_top, h_bottom, h_mid, h_left, h_right));
      _mm256_storeu_pd(d2 + col, kernel(h_mid, b_bottom, h_bottom, b_left, b_right));
    };
    std::size_t col = 0;
    for (; col + 4 <= cols - 1; col += 4) block(col);
    // handle mispadding with another full computation to avoid unecessary special handling
    if (col < cols - 1) block(cols - 5);
    // copy padding in main loop while in cache
    d1[0] = r1[0]; d1[cols-1] = r1[cols-1];
    d2[0] = r2[0]; d2[cols-1] = r2[cols-1];
  };

  if (rows >= 3 && cols >= 5) {
    const std::size_t interior = rows - 2, pairs = interior / 2;

    #pragma omp parallel for schedule(static)
    for (std::size_t p=0; p<pairs; ++p) row_pair(1 + 2*p);

    if (interior & 1) {
      if (rows >= 4) row_pair(rows - 3);
      else for (std::size_t col=1; col<cols-1; ++col)
        dst[cols+col] = c_kernel(cells, 1, col, cols);
    }
  } else if (rows >= 3 && cols >= 3) {
    for (std::size_t row=1; row<rows-1; ++row) for (std::size_t col=1; col<cols-1; ++col)
      dst[row*cols+col] = c_kernel(cells, row, col, cols);
  }

  memcpy(dst, cells, cols * sizeof(double));
  memcpy(dst+(rows-1)*cols, cells+(rows-1)*cols, cols * sizeof(double));
}

void apply_stencil(const Grid& __restrict__ old_grid, Grid& __restrict__ new_grid) {
  return stencil_impl(new_grid.get_rows(), new_grid.get_cols(), old_grid.data(), new_grid.data());
}
