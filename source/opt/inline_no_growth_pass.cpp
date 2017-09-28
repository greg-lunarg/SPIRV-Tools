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

void InlineNoGrowthPass::ComputeCallSize() {
  funcId2callSize_.clear();
  for (auto& fn : *module_) {
    uint32_t icnt = 0;
    fn.ForEachParam([&icnt, this](const ir::Instruction* inst) {
      (void) inst;
      ++icnt;
    });
    // If the param count is > 9, the size of the OpFunctionCall instruction
    // is the size of two OpFAdds, so increment the total size of the call.
    if (icnt > 9) ++icnt;
    funcId2callSize_[fn.result_id()] = icnt;
  }
}

void InlineNoGrowthPass::ComputeInlinedSize() {
  funcId2inlinedSize_.clear();
  for (auto& fn : *module_) {
    uint32_t icnt = 0;
    fn.ForEachInst([&icnt, this](ir::Instruction* inst) {
      switch(inst->opcode()) {
        case SpvOpFunction:
        case SpvOpFunctionParameter:
        case SpvOpVariable:
        case SpvOpLabel:
        case SpvOpLoad:
        case SpvOpStore:
        case SpvOpAccessChain:
        case SpvOpReturn:
        case SpvOpReturnValue:
        case SpvOpFunctionEnd:
          // (Likely) Removed by inlining or optimization
          // TODO(greg-lunarg): Count non-constant index OpAccessChain/OpLoad 
          // TODO(greg-lunarg): Count OpStore to buffer
          break;
        case SpvOpFunctionCall: {
          // If we are looking at inlined size we know that the function
          // is called more than once, so we know its callees will also
          // be called more than once, so will only be inlined based on
          // size. So we know the amount of code resulting from the call will
          // be no bigger (and likely not much smaller) than the size of the
          // actual call.
          uint32_t calleeId = inst->GetSingleWordInOperand(0);
          icnt += funcId2callSize_[calleeId];
          } break;
        default:
          ++icnt;
          break;
      }
    });
    funcId2inlinedSize_[fn.result_id()] = icnt;
  }
}

bool InlineNoGrowthPass::IsNoGrowthCall(const ir::Instruction* callInst) {
  const uint32_t calleeId = callInst->GetSingleWordInOperand(0);
  // Functions with only one call are no-growth because the original will
  // be DCE'd.
  if (funcId2callCount_[calleeId] == 1) return true;
  // Functions whose inlined size is smaller than the call size are no-growth
  return funcId2inlinedSize_[calleeId] < funcId2callSize_[calleeId];
}

bool InlineNoGrowthPass::InlineNoGrowth(ir::Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (IsInlinableFunctionCall(&*ii) && IsNoGrowthCall(&*ii)) {
        // Save inlinee id for call count update
        uint32_t inlineeId = ii->GetSingleWordInOperand(0);
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
        UpdateCallDataAfterInlining(inlineeId);
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
  // Must preceed ComputeInlinedSize call
  ComputeCallSize();
  ComputeInlinedSize();
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
