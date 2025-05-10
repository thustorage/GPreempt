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

#include "client.h"

#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <json/json.h>

namespace DISB
{

using std::chrono::system_clock;

Client::Client():
    name("Unknown"),
    basicAnalyzer(std::make_shared<BasicAnalyzer>())
{
    basicAnalyzer->setClient(this);
}

std::string Client::getName()
{
    return name;
}

void Client::setName(const std::string &_name)
{
    name = _name;
}

std::vector<std::shared_ptr<Record>> Client::produceRecords()
{
    std::shared_ptr<TimePoints> timePoints = std::make_shared<TimePoints>();
    std::vector<std::shared_ptr<Record>> records;

    auto record = basicAnalyzer->produceRecordWrapper();
    record->timePoints = timePoints;
    records.push_back(record);

    for (auto customAnalyzer : customAnalyzers) {
        auto customRecord = customAnalyzer->produceRecordWrapper();
        customRecord->timePoints = timePoints;
        records.push_back(customRecord);
    }

    return records;
}

Json::Value Client::generateReport()
{
    Json::Value analyzerReports;
    
    analyzerReports.append(basicAnalyzer->generateReport());
    for (auto customAnalyzer : customAnalyzers) {
        analyzerReports.append(customAnalyzer->generateReport());
    }

    Json::Value report;
    report["clientName"] = name;
    report["analyzers"] = analyzerReports;
    return report;
}

void Client::addAnalyzer(std::shared_ptr<Analyzer> customAnalyzer)
{
    customAnalyzer->setClient(this);
    customAnalyzers.push_back(customAnalyzer);
}

void Client::initAnalyzers()
{
    basicAnalyzer->init();
    for (auto customAnalyzer : customAnalyzers) {
        customAnalyzer->init();
    }
}

void Client::startAnalyzers(const std::chrono::system_clock::time_point &beginTime)
{
    basicAnalyzer->start(beginTime);
    for (auto customAnalyzer : customAnalyzers) {
        customAnalyzer->start(beginTime);
    }
}

void Client::stopAnalyzers(const std::chrono::system_clock::time_point &endTime)
{
    basicAnalyzer->stop(endTime);
    for (auto customAnalyzer : customAnalyzers) {
        customAnalyzer->stop(endTime);
    }
}

void Client::setStandAloneLatency(std::chrono::nanoseconds standAloneLatency)
{
    basicAnalyzer->setStandAloneLatency(standAloneLatency);
}

std::function<void(std::atomic<bool>&, int)> daemonThread = nullptr;

void setDaemonThread(std::function<void(std::atomic<bool>&, int)> func)
{
    daemonThread = func;
}

} // namespace DISB
