#include "parReader.hpp"

ParRead::ParRead(std::string &inifile) {

  read(inifile);

}

void ParRead::read(std::string &inifile) {

  std::ifstream infile(inifile);

  int counter = -1;

  std::string opt = "";

  std::string line;
  while(std::getline(infile, line)) {

    if(line.rfind("[[", 0) == 0) {

      line.replace(0, 2, "");
      size_t pos = line.find(" OPTIONS]]");
      line.replace(pos, 10, "");
      std::transform(line.begin(), line.end(), line.begin(), ::tolower);
      ++counter;
      optionsBenchmarks.push_back(line);
      _options.push_back(setupAide());
      options.push_back(std::vector<setupAide>());
      continue;

    }

    if(line.rfind("[", 0) == 0) {
      line.replace(0, 1, "");
      size_t pos = line.find("]");
      line.replace(pos, 2, "");
      opt = line;
      continue;
    }

    if(line != "" && line.rfind("#", 0) != 0) {

      if(counter == -1)
        generalOptions.setArgs(opt, line);
      else
        _options[counter].setArgs(opt, line);

    }

  }

  for(size_t i = 0; i < _options.size(); ++i) {
    std::vector<std::string> processed;
    setupAide opt;
    // add general options
    std::vector<std::string> keys = generalOptions.getKeyword();
    for(size_t j = 0; j < keys.size(); ++j) {
      opt.setArgs(keys[j], generalOptions.getArgs(keys[j]));
    }
    generateOptions(_options[i], opt, processed, i);
  }

}

void ParRead::generateOptions(setupAide &inOpt, setupAide outOpt, std::vector<std::string> processed, int benchIndex) {

  std::vector<std::string> inKey = inOpt.getKeyword();
  for(size_t i = 0; i < inKey.size(); ++i) {
    // see if this one is already processed
    std::vector<std::string>::iterator it = std::find(processed.begin(), processed.end(), inKey[i]);
    // not yet:
    if(it == processed.end()) {

      std::vector<std::string> parts = explode(inOpt.getArgs(inKey[i]), *",");

      processed.push_back(inKey[i]);

      for(size_t j = 0; j < parts.size(); ++j) {

        outOpt.setArgs(inKey[i], parts[j]);
        generateOptions(inOpt, outOpt, processed, benchIndex);

      }

      return;

    }

  }

  options[benchIndex].push_back(outOpt);

}

const std::vector<std::string> ParRead::explode(const std::string& s, const char& c) {
  std::string buff{""};
  std::vector<std::string> v;

  for(auto n:s) {
    if(n != c) buff+=n; else
    if(n == c && buff != "") { v.push_back(buff); buff = ""; }
  }
  if(buff != "") v.push_back(buff);

  return v;
}

std::vector<setupAide> &ParRead::getOptions(std::string benchmark) {

  std::transform(benchmark.begin(), benchmark.end(), benchmark.begin(), ::tolower);

  std::vector<std::string>::iterator it = std::find(optionsBenchmarks.begin(), optionsBenchmarks.end(), benchmark);

  if(it == optionsBenchmarks.end()) {
    std::cout << "ERROR: No options found for benchmark " << benchmark << std::endl;
    return options[0];
  }

  int index = std::distance(optionsBenchmarks.begin(), it);

  return options[index];

}
