// Minimal CLI flag parser shared by serial/openmp/mpi mains.
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdlib>

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) continue;
            std::string key = a.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                flags_[key] = argv[i + 1];
                i++;
            } else {
                flags_[key] = "1"; // boolean flag
            }
        }
    }

    bool has(const std::string& key) const { return flags_.count(key) > 0; }

    std::string get(const std::string& key, const std::string& def = "") const {
        auto it = flags_.find(key);
        return it == flags_.end() ? def : it->second;
    }

    int get_int(const std::string& key, int def) const {
        auto it = flags_.find(key);
        return it == flags_.end() ? def : std::atoi(it->second.c_str());
    }

    double get_double(const std::string& key, double def) const {
        auto it = flags_.find(key);
        return it == flags_.end() ? def : std::atof(it->second.c_str());
    }

private:
    std::unordered_map<std::string, std::string> flags_;
};
