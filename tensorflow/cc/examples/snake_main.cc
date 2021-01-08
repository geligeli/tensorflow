#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/time/time.h"

ABSL_FLAG(bool, big_menu, true,
          "Include 'advanced' options in the menu listing");
ABSL_FLAG(std::string, output_dir, "foo/bar/baz/", "output file dir");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::vector<std::string> v = {"foo", "bar", "baz"};
  std::string s = absl::StrJoin(v, "-");

  std::cout << "Joined string: " << s << "\n";

  return 0;
}