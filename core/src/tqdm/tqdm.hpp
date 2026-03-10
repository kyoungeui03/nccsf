#ifndef CSF_GRF_TQDM_STUB_HPP
#define CSF_GRF_TQDM_STUB_HPP

#include <cstddef>
#include <string>

namespace tq {

class progress_bar {
public:
  progress_bar() = default;

  void set_theme_line_color(const std::string&) {}
  void set_theme_bar_color(const std::string&) {}
  void set_theme_spin_color(const std::string&) {}
  void set_theme_progress_color(const std::string&) {}
  void set_prefix(const std::string&) {}
  void set_bar_size(std::size_t) {}
  void progress(std::size_t, std::size_t) {}
  void finish() {}
};

}  // namespace tq

#endif
