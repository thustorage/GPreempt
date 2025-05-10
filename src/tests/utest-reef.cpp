#include "executor.h"
#include <vector>
#include <assert.h>
#include "reef_scheduler.h"

#define PATH_OF(name) MODEL_PATH "/" #name

// int main(int argc, char **argv) {
//     if(argc < 2) {
//         printf("Usage: %s <model>\n", argv[0]);
//         return 1;
//     }
//     foo::util::init_cuda();
//     foo::BaseExecutor executor;
//     GPUstream stream;
//     GPUStreamCreate(&stream, 0);
//     std::string model_name = argv[1];
//     CHECK_STATUS(executor.load_model(std::string(MODEL_PATH) + "/" + model_name));
//     CHECK_STATUS(executor.load_param(std::string(MODEL_PATH) + "/" + model_name));
//     size_t input_size = executor.get_data_size("data");;
//     size_t output_size = executor.get_data_size("heads");
//     if(model_name == "bert") {
//         std::vector<int64_t> input({101, 7592, 1010, 2088,  999,  102});
//         CHECK_STATUS(executor.set_input("data", input.data(), input_size, stream));
//     } else {
//         std::vector<float> input(input_size / sizeof(float));
//         for(size_t i = 0; i < input.size(); i++) {
//             input[i] = 10.0;
//         }
//         CHECK_STATUS(executor.set_input("data", input.data(), input_size, stream));
//     }
//     CHECK_STATUS(executor.execute(stream));
//     std::vector<float> output(output_size / sizeof(float));
//     CHECK_STATUS(executor.get_output("heads", output, stream));
//     GPUStreamSynchronize(stream);
//     for(size_t i = 0; i < 10; i++) {
//         printf("%f\n", output[i]);
//     }
//     return 0;
// }

reef::executor::REEFScheduler *scheduler;

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: %s <model>\n", argv[0]);
        return 1;
    }
    // foo::util::init_cuda();
    // foo::BaseExecutor executor;
    // GPUstream stream;
    // GPUStreamCreate(&stream, 0);
    // std::string model_name = argv[1];
    // CHECK_STATUS(executor.load_model(std::string(MODEL_PATH) + "/" + model_name));
    // CHECK_STATUS(executor.load_param(std::string(MODEL_PATH) + "/" + model_name));
    scheduler = new reef::executor::REEFScheduler();
    scheduler->run();
    reef::executor::REEFScheduler::ModelID modelID;
    std::string model_name = argv[1];
    CHECK_STATUS(scheduler->load_model(std::string(MODEL_PATH) + "/" + model_name, modelID));
    reef::executor::REEFScheduler::QueueID qid;
    // scheduler->create_queue(reef::executor::REEFScheduler::TaskQueueType::BestEffortQueue, qid);
    // scheduler->bind_model_queue(qid, modelID);

    size_t input_size;
    scheduler->get_data_size(modelID, "data", input_size);
    size_t output_size;
    scheduler->get_data_size(modelID, "heads", output_size);
    printf("input size: %ld\n", input_size);
    printf("output size: %ld\n", output_size);

    if(model_name == "bert") {
        std::vector<int64_t> input({101, 7592, 1010, 2088,  999,  102});
        // CHECK_STATUS(executor.set_input("data", input.data(), input_size, stream));
        scheduler->set_input(modelID, "data", input.data(), input_size);
    } else {
        std::vector<float> input(input_size / sizeof(float));
        for(size_t i = 0; i < input.size(); i++) {
            input[i] = 10.0;
        }
        // CHECK_STATUS(executor.set_input("data", input.data(), input_size, stream));
        scheduler->set_input(modelID, "data", input.data(), input_size);
    }
    // CHECK_STATUS(executor.execute(stream));
    reef::executor::REEFScheduler::TaskID tid;
    scheduler->new_task(modelID, tid);
    scheduler->wait_task(tid);
    
    std::vector<float> output(output_size / sizeof(float));
    scheduler->get_output(modelID, "heads", output.data(), output_size);
    // CHECK_STATUS(executor.get_output("heads", output, stream));
    // GPUStreamSynchronize(stream);
    for(size_t i = 0; i < 10; i++) {
        printf("%f\n", output[i]);
    }
    return 0;
}

