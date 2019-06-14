// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "convert_to_half_pass.h"

#include "source/opt/ir_builder.h"

namespace {

// Indices of operands in SPIR-V instructions
  static const int kEntryPointFunctionIdInIdx = 1;

}  // anonymous namespace

namespace spvtools {
namespace opt {

void ConvertToHalfPass::GenHalfCode(
    Instruction* inst,
    InstructionList* new_insts) {
}

bool ConvertToHalfPass::ProcessFunction(Function* func) {
  bool modified = false;
  InstructionList new_insts;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      // Generate new instructions if warranted
      GenHalfCode(&*ii, &new_insts);
      if (new_insts.empty())
        continue;
      // Insert new instructions and replace original result
      // id with new id
      uint32_t nid = new_insts.back().result_id();
      uint32_t oid = ii->result_id();
      (void) ii.MoveBefore(&new_insts);
      context()->ReplaceAllUsesWith(oid, nid);
      modified = true;
    }
  }
  return modified;
}

bool ConvertToHalfPass::ProcessCallTreeFromRoots(
    std::queue<uint32_t>* roots) {
  bool modified = false;
  std::unordered_set<uint32_t> done;
  // Process all functions from roots
  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = id2function_.at(fi);
      // Add calls first so we don't add new output function
      context()->AddCalls(fn, roots);
      modified = ProcessFunction(fn) || modified;
    }
  }
  return modified;
}

Pass::Status ConvertToHalfPass::ProcessImpl() {
  // Add together the roots of all entry points
  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  bool modified = ProcessCallTreeFromRoots(&roots);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status ConvertToHalfPass::Process() {
  Initialize();
  return ProcessImpl();
}

void ConvertToHalfPass::Initialize() {
  id2function_.clear();
  for (auto& fn : *get_module()) {
    id2function_[fn.result_id()] = &fn;
  }
}

}  // namespace opt
}  // namespace spvtools
