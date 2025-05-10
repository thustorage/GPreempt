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

#ifndef _DISB_UTILS_H_
#define _DISB_UTILS_H_

#include <string>
#include <chrono>
#include <json/json.h>

#define THOUSAND 1000
#define BILLION 1000000000

std::string readStringFromFile(const std::string &filename);
Json::Value readJsonFromFile(const std::string &filename);
void writeJsonToFile(const std::string &filename, const Json::Value &json);
std::string joinPath(std::string path1, std::string path2);
std::string timepointToString(const std::chrono::system_clock::time_point &timepoint);

#endif
