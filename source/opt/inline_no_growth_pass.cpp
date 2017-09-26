// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "inline_no_growth_pass.h"

namespace spvtools {
namespace opt {

namespace {

} // anonymous namespace

bool InlineNoGrowthPass::IsNoGrowthCall(const ir::Instruction* callInst) {
  (void) callInst;
  return true;
}

bool InlineNoGrowthPass::InlineNoGrowth(ir::Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (IsInlinableFunctionCall(&*ii) && IsNoGrowthCall(&*ii)) {
        // Save callee id for call count update
        uint32_t calleeId = ii->GetSingleWordInOperand(0);
        // Inline call.
        std::vector<std::unique_ptr<ir::BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<ir::Instruction>> newVars;
        GenInlineCode(&newBlocks, &newVars, ii, bi);
        // If call block is replaced with more than one block, point
        // succeeding phis at new last block.
        if (newBlocks.size() > 1)
          UpdateSucceedingPhis(newBlocks);
        // Replace old calling block with new block(s).
        bi = bi.Erase();
        bi = bi.InsertBefore(&newBlocks);
        // Insert new function variables.
        if (newVars.size() > 0) func->begin()->begin().InsertBefore(&newVars);
        // Update call data
        (void) calleeId;
        // Restart inlining at beginning of calling block.
        ii = bi->begin();
        modified = true;
      } else {
        ++ii;
      }
    }
  }
  return modified;
}

void InlineNoGrowthPass::Initialize(ir::Module* module) {
  InitializeInline(module);
};

Pass::Status InlineNoGrowthPass::ProcessImpl() {
  // Do opaque inlining on each function in entry point call tree
  ProcessFunction pfn = [this](ir::Function* fp) {
    return InlineNoGrowth(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, module_);
  FinalizeNextId(module_);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InlineNoGrowthPass::InlineNoGrowthPass() {}

Pass::Status InlineNoGrowthPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
