#if !defined(driver_parreader_hpp_)
#define driver_parreader_hpp_

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mpi.h>
#include "setupAide.hpp"

class IniReader {

private:
  std::vector<setupAide> _options;  // the unprocessed options
  std::vector<std::vector<setupAide> > options; // the processed options
  std::vector<std::string> optionsBenchmarks; // the name of the benchmarks, in order
  setupAide generalOptions;

  void read(std::string &inifile);
  void generateOptions(setupAide &inOpt, setupAide outOpt, std::vector<std::string> processed, int benchIndex);
  const std::vector<std::string> explode(const std::string& s, const char& c);

public:
  IniReader(std::string &inifile);
  std::vector<setupAide> &getOptions(std::string benchmark);

};

#endif
