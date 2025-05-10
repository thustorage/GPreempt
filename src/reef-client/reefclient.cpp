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
// # - Extract the reef scheduler from the original file.
// # - Remove dynamic kernel padding.
// # 
// # All modifications are also licensed under the Apache License, Version 2.0.
// # ------------------------------------------------------------------------------


#include "disb.h"

#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <assert.h>
#include <iostream>
#include <json/json.h>
#include "executor.h"

#include "reef_scheduler.h"

#define PATH_OF(name) MODEL_PATH "/" #name

// Reset based version
// Modified from reef

reef::executor::REEFScheduler *scheduler;

std::atomic_int g_model_id(0);

class FooClient: public DISB::DependentClient
{
private:
    reef::executor::REEFScheduler::QueueID qid;

public:
    FooClient(const Json::Value &config) {
        model_name = config["model_name"].asString();
        priority = config["priority"].asInt();
        preprocess_time = config["preprocess_time"].asUInt64();
        setName(config["name"].asString());
    }
    
    ~FooClient() {}

    virtual void init() override
    {
        Status s = scheduler->create_queue(this->priority == 0 ? 
            reef::executor::REEFScheduler::TaskQueueType::RealTimeQueue
            : reef::executor::REEFScheduler::TaskQueueType::BestEffortQueue, 
           qid
        );
        if (s != Status::Succ) {
            std::cerr << "Failed to create queue" << std::endl;
            exit(-1);
        }

        s = scheduler->load_model(std::string(MODEL_PATH) + "/" + model_name, modelID);
        if(s != Status::Succ) {
            std::cerr << "Failed to load model" << std::endl;
            exit(-1);
        }
        s = scheduler->bind_model_queue(qid, modelID);
        if(s != Status::Succ) {
            std::cerr << "Failed to bind model to queue" << std::endl;
            exit(-1);
        }
        
        scheduler->get_data_size(modelID, "data", input_size);
        scheduler->get_data_size(modelID, "heads", output_size);
        // scheduler->create_queue(const TaskQueueType &qtp, QueueID &qid)
    }
    
    virtual void prepareInput() override {}

    virtual void preprocess() override {
        usleep(preprocess_time);
    }

    virtual void copyInput() override
    {
        // void *data = malloc(input_size);
        // scheduler->set_input(modelID, "data", data, input_size);
        // free(data);
    }

    virtual void infer() override
    {
        reef::executor::REEFScheduler::TaskID tid;
        scheduler->new_task(modelID, tid);
        scheduler->wait_task(tid);
    }

    virtual void copyOutput() override
    {
        // void *output = malloc(output_size);
        // scheduler->get_output(modelID, "heads", output, output_size);
        // free(output);
    }

    virtual void postprocess() override {}
private:
    typedef uint32_t ModelID;
    ModelID modelID;
    std::string model_name;
    int priority;
    size_t input_size;
    size_t output_size;
    uint64_t preprocess_time;
};

std::shared_ptr<DISB::Client> FooClientFactory(const Json::Value &config)
{
    return std::make_shared<FooClient>(config);
}



int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return -1;
    }
    scheduler = new reef::executor::REEFScheduler();
    scheduler->run();
    DISB::BenchmarkSuite benchmark;
    std::string jsonStr = readStringFromFile(argv[1]);
    benchmark.init(jsonStr, FooClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << report << std::endl;
    scheduler->shutdown();
    delete scheduler;
    return 0;
}
