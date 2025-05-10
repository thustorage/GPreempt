// # Original Copyright 2023 SJTU-IPADS
// #
// # Licensed under the Apache License, Version 2.0 (the "License");
// # you may not use this file except in compliance with the License.
// # You may obtain a copy of the License at
// #
// #     http://www.apache.org/licenses/LICENSE-2.0
// #
// # Unless required by applicable law or agreed to in writing, software
// # distributed under the License is distributed on an "AS IS" BASIS,
// # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// # See the License for the specific language governing permissions and
// # limitations under the License.
// #
// # ------------------------------------------------------------------------------
// # MODIFICATIONS:
// # This file has been modified in 2025.
// // #
// # The following changes were made to the original file:
// # - Added a new function to set the daemon thread.
// # - Modified the third-party library path to include a new version of jsoncpp.
// #
// # All modifications are also licensed under the Apache License, Version 2.0.
// # ------------------------------------------------------------------------------

#ifndef _DISB_BENCHMARK_H_
#define _DISB_BENCHMARK_H_

#include "analyzer.h"
#include "load.h"
#include "client.h"

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <json/json.h>

#define WARMUP_TIME                         10
#define STANDALONE_LATENCY_MEASURE_TIME     100
#define DELAY_BEGIN_SECONDS                 4

namespace DISB
{

struct BenchmarkTask
{
    bool isDependent = false;
    std::string id;
    std::vector<BenchmarkTask *> nextTasks;
    std::shared_ptr<Client> client;
    std::shared_ptr<Load> load;
    std::shared_ptr<StandAloneLatency> standAloneLatency;

    BenchmarkTask(): id("UNKNOWN") {}
    BenchmarkTask(std::string taskId, std::shared_ptr<Client> clt, std::shared_ptr<Load> stg);
    
    void inferOnce();
    void runBenchmark(std::chrono::system_clock::time_point beginTime,
                      std::chrono::system_clock::time_point endTime);
};

class BenchmarkSuite
{
private:
    std::chrono::seconds benchmarkTime;
    std::chrono::system_clock::time_point beginTime;
    std::chrono::system_clock::time_point endTime;
    std::map<std::string, BenchmarkTask> tasks;

public:
    BenchmarkSuite();
    ~BenchmarkSuite();

    void init(const std::string &configJsonStr,
              std::shared_ptr<Client> clientFactory(const Json::Value &config),
              std::shared_ptr<Load> loadFactory(const Json::Value &config) = builtinLoadFactory);
    
    void init(const Json::Value &configJson,
              std::shared_ptr<Client> clientFactory(const Json::Value &config),
              std::shared_ptr<Load> loadFactory(const Json::Value &config) = builtinLoadFactory);

    void run(void loadCoordinator(const std::vector<LoadInfo> &loadInfos) = builtinLoadCoordinator);
    Json::Value generateReport();
};

} // namespace DISB

#endif
