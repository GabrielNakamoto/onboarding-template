#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <cstring>

template <typename T>
class View {
  private: 
    T *m_buffer;
  public:
    const std::size_t m_rows;
    const std::size_t m_cols;

    View(std::size_t rows, std::size_t cols, T *ptr) : m_rows(rows), m_cols(cols), m_buffer(ptr) {}

    T       *get()        { return m_buffer; }
    const T *get() const  { return m_buffer; }
};

class Grid {
private:
  struct FreeDeleter {
    void operator()(double* p) const noexcept { std::free(p); }
  };

  static constexpr std::size_t kAlign = 64;

  const std::size_t m_rows;
  const std::size_t m_cols;
  std::unique_ptr<double[], FreeDeleter> m_buffer;

public:
  Grid(std::size_t rows, std::size_t cols) : m_rows(rows), m_cols(cols) {
    std::size_t bytes = (rows * cols * sizeof(double) + kAlign - 1) / kAlign * kAlign;
    void* p = std::aligned_alloc(kAlign, bytes);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, bytes);
    m_buffer = std::unique_ptr<double[], FreeDeleter>{static_cast<double*>(p)};
  }

  double& operator()(std::size_t i, std::size_t j)       { return m_buffer[i*m_cols + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return m_buffer[i*m_cols + j]; }

  View<double> view()             { return View<double>(m_rows, m_cols, m_buffer.get()); }
  const View<double> view() const { return View<double>(m_rows, m_cols, m_buffer.get()); }
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

static void stencil_impl(const View<double> old_buffer, View<double> new_buffer) {
  const double *__restrict__ old_data = old_buffer.get();
  double *__restrict__ new_data = new_buffer.get();
  const std::size_t cols = new_buffer.m_cols, rows = new_buffer.m_rows;
  auto row_pair = [=](std::size_t row) {
    const double *r0 = old_data + (row-1)*cols, *r1 = r0 + cols, *r2 = r1 + cols, *r3 = r2 + cols;
    double *d1 = new_data + row*cols, *d2 = d1 + cols;
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
    // copy padding in main loop to take advantage of cache locality
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
        new_buffer.get()[cols+col] = c_kernel(old_buffer.get(), 1, col, cols);
    }
  } else if (rows >= 3 && cols >= 3) {
    for (std::size_t row=1; row<rows-1; ++row) for (std::size_t col=1; col<cols-1; ++col)
      new_buffer.get()[row*cols+col] = c_kernel(old_buffer.get(), row, col, cols);
  }

  memcpy(new_buffer.get(), old_buffer.get(), cols * sizeof(double));
  memcpy(new_buffer.get()+(rows-1)*cols, old_buffer.get()+(rows-1)*cols, cols * sizeof(double));
}

void apply_stencil(const Grid& __restrict__ old_grid, Grid& __restrict__ new_grid) {
  return stencil_impl(old_grid.view(), new_grid.view());
}
