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

#ifndef _DISB_CLIENT_H_
#define _DISB_CLIENT_H_

#include "analyzer.h"

#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <json/json.h>
#include <functional>

namespace DISB
{

class InferResult
{
public:
    int64_t resultSerialNumber = 0;
    std::string producerTaskId;
    std::string consumerTaskId;
};

class Client
{
private:
    std::string name;
    std::shared_ptr<BasicAnalyzer> basicAnalyzer;
    std::vector<std::shared_ptr<Analyzer>> customAnalyzers;

public:
    Client();
    std::string getName();
    void setName(const std::string &name);

    std::vector<std::shared_ptr<Record>> produceRecords();
    Json::Value generateReport();
    void addAnalyzer(std::shared_ptr<Analyzer> customAnalyzer);
    void initAnalyzers();
    void startAnalyzers(const std::chrono::system_clock::time_point &beginTime);
    void stopAnalyzers(const std::chrono::system_clock::time_point &endTime);
    void setStandAloneLatency(std::chrono::nanoseconds standAloneLatency);

    virtual void init() {}
    virtual void initInThread() {}
    virtual void prepareInput() {}
    virtual void preprocess() {}
    virtual void copyInput() {}
    virtual void infer() {}
    virtual void copyOutput() {}
    virtual void postprocessNoRecord() {}
    virtual void postprocess() {}
    virtual std::shared_ptr<InferResult> produceResult() { return std::make_shared<InferResult>(); };
};

extern std::function<void(std::atomic<bool>&, int)> daemonThread;

void setDaemonThread(std::function<void(std::atomic<bool>&, int)> func);

class DependentClient: public Client
{
public:
    virtual void consumePrevResults(const std::map<std::string, std::shared_ptr<InferResult>> &prevResults) {}
    virtual std::map<std::string, std::shared_ptr<InferResult>> produceDummyPrevResults()
    {
        return std::map<std::string, std::shared_ptr<InferResult>>();
    }
};

} // namespace DISB

#endif
