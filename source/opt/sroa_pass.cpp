// Copyright (c) 2016 The Khronos Group Inc.
// Copyright (c) 2016 Valve Corporation
// Copyright (c) 2016 LunarG Inc.
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

#include "sroa_pass.h"
#include "iterator.h"

#include <unordered_map>

#define SPV_ENTRY_POINT_FUNCTION_ID 1
#define SPV_FUNCTION_CALL_FUNCTION_ID 2
#define SPV_FUNCTION_CALL_ARGUMENT_ID 3
#define SPV_FUNCTION_PARAMETER_RESULT_ID 1
#define SPV_STORE_OBJECT_ID 1
#define SPV_RETURN_VALUE_ID 0
#define SPV_TYPE_POINTER_STORAGE_CLASS 1
#define SPV_TYPE_POINTER_TYPE_ID 2

namespace spvtools {
namespace opt {

bool SRoAPass::IsStructOfScalar(ir::Instruction* typeInst) {
  if (typeInst->opcode() != SpvOpTypeStruct)
    return false;
  size_t aggCount = 0;
  typeInst->ForEachInId([&aggCount,this](uint32_t* idp) {
    auto compType = def_use_mgr_->GetDef(*idp);
    auto typeOp = compType->opcode();
    switch (typeOp) {
      case SpvOpTypeFloat:
      case SpvOpTypeVector:
      case SpvOpTypeInt:
      case SpvOpTypeMatrix:
      case SpvOpTypeBool:
        break;
      default:
        aggCount++;
        break;
    }
  });
  return aggCount == 0;
}

bool SRoAPass::SRoA(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
        switch (ii->opcode()) { 
          case SpvOpLoad: {
            // Identify leading load of load/store pair(s) of targeted struct type
            auto typeInst = def_use_mgr_->GetDef(ii->type_id());
            if (!IsStructOfScalar(typeInst))
              break;
            size_t nonStoreUse = 0;
            if (analysis::UseList* uses = def_use_mgr_->GetUses(ii->result_id())) {
                nonStoreUse =
                    std::count_if(uses->begin(), uses->end(), [](const analysis::Use& u) {
                    return (u.inst->opcode() != SpvOpStore);
                });
            }
            if (nonStoreUse > 0)
              break;
          } break;
          case SpvOpStore: {
          } break;
      }
    }
  }
  return modified;
}

void SRoAPass::Initialize(ir::Module* module) {
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

  // Initialize next unused Id
  nextId_ = 0;
  for (const auto& id_def : def_use_mgr_->id_to_defs()) {
    nextId_ = std::max(nextId_, id_def.first);
  }
  nextId_++;

  module_ = module;

  // initialize function and block maps
  id2function.clear();
  id2block.clear();
  for (auto& fn : *module_) {
    id2function[fn.GetResultId()] = &fn;
    for (auto& blk : fn) {
      id2block[blk.GetLabelId()] = &blk;
    }
  }
};

Pass::Status SRoAPass::ProcessImpl() {
  // do exhaustive inlining on each entry point function in module
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function[e.GetOperand(SPV_ENTRY_POINT_FUNCTION_ID).words[0]];
    modified = modified || SRoA(fn);
  }

  finalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

SRoAPass::SRoAPass()
    : nextId_(0), module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status SRoAPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
