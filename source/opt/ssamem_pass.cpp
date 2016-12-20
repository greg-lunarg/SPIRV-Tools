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

#include "ssamem_pass.h"
#include "iterator.h"

/*
#define SPV_FUNCTION_CALL_FUNCTION_ID 2
#define SPV_FUNCTION_CALL_ARGUMENT_ID 3
#define SPV_FUNCTION_PARAMETER_RESULT_ID 1
#define SPV_STORE_POINTER_ID 0
#define SPV_STORE_OBJECT_ID 1
#define SPV_RETURN_VALUE_ID 0
#define SPV_TYPE_POINTER_STORAGE_CLASS 1
#define SPV_TYPE_POINTER_TYPE_ID 2
*/

static const int SPV_ENTRY_POINT_FUNCTION_ID = 1;
static const int SPV_STORE_POINTER_ID = 0;
static const int SPV_STORE_VAL_ID = 1;
static const int SPV_ACCESS_CHAIN_PTR_ID = 0;
static const int SPV_ACCESS_CHAIN_IDX0_ID = 1;
static const int SPV_TYPE_PTR_STORAGE_CLASS = 0;
static const int SPV_TYPE_PTR_TYPE_ID = 1;
static const int SPV_LOAD_PTR_ID = 0;

namespace spvtools {
namespace opt {

bool SSAMemPass::isMathType(ir::Instruction* typeInst) {
  switch (typeInst->opcode()) {
  case SpvOpTypeInt:
  case SpvOpTypeFloat:
  case SpvOpTypeBool:
  case SpvOpTypeVector:
  case SpvOpTypeMatrix:
    return true;
  default:
    break;
  }
  return false;
}

bool SSAMemPass::isTargetType(ir::Instruction* typeInst) {
  if (isMathType(typeInst))
    return true;
  if (typeInst->opcode() != SpvOpTypeStruct)
    return false;
  int nonMathComp = 0;
  typeInst->ForEachInId([&nonMathComp,this](uint32_t* tid) {
    ir::Instruction* typeInst =
        def_use_mgr_->id_to_defs().find(*tid)->second;
    if (!isMathType(typeInst)) nonMathComp++;
  });
  return nonMathComp == 0;
}

void SSAMemPass::SSAMemAnalyze(ir::Function* func) {
  ssaVars.clear();
  ssaComps.clear();
  ssaCompVars.clear();
  nonSsaVars.clear();
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      if (ii->opcode() != SpvOpStore)
        continue;
      const uint32_t ptrId = ii->GetInOperand(SPV_STORE_POINTER_ID).words[0];
      const ir::Instruction* ptrInst =
          def_use_mgr_->id_to_defs().find(ptrId)->second;
      const uint32_t varId = ptrInst->opcode() == SpvOpAccessChain ?
          ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0] :
          ptrId;

      // Verify store variable is target type
      const ir::Instruction* varInst =
          def_use_mgr_->id_to_defs().find(varId)->second;
      const uint32_t varTypeId = varInst->type_id();
      if (seenNonTargetTypes.find(varTypeId) != seenNonTargetTypes.end()) {
        nonSsaVars.insert(varId);
        continue;
      }
      if (seenTargetTypes.find(varTypeId) == seenTargetTypes.end()) {
        const ir::Instruction* varTypeInst =
            def_use_mgr_->id_to_defs().find(varTypeId)->second;
        if (varTypeInst->GetInOperand(SPV_TYPE_PTR_STORAGE_CLASS).words[0] !=
            SpvStorageClassFunction) {
          seenNonTargetTypes.insert(varTypeId);
          nonSsaVars.insert(varId);
          continue;
        }
        const uint32_t varPteTypeId =
            varTypeInst->GetInOperand(SPV_TYPE_PTR_TYPE_ID).words[0];
        ir::Instruction* varPteTypeInst =
            def_use_mgr_->id_to_defs().find(varPteTypeId)->second;
        if (!isTargetType(varPteTypeInst)) {
          seenNonTargetTypes.insert(varTypeId);
          nonSsaVars.insert(varId);
          continue;
        }
        if (ptrInst->opcode() == SpvOpAccessChain &&
            isMathType(varPteTypeInst)) {
          seenNonTargetTypes.insert(varTypeId);
          nonSsaVars.insert(varId);
          continue;
        }
        seenTargetTypes.insert(varTypeId);
      }

      // Verify store variable/component is not yet assigned.
      // If whole variable is stored, there should not be any
      // component stores and vice-versa.
      if (nonSsaVars.find(varId) != nonSsaVars.end())
        continue;
      if (ssaVars.find(varId) != ssaVars.end()) {
        nonSsaVars.insert(varId);
        continue;
      }
      if (ptrInst->opcode() == SpvOpAccessChain) {
        if (ptrInst->NumInOperands() != 2) {
          nonSsaVars.insert(varId);
          continue;
        }
        const uint32_t idxId =
            ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
        if (ssaComps.find(std::make_pair(varId, idxId)) != ssaComps.end()) {
          nonSsaVars.insert(varId);
          continue;
        }
        ssaCompVars.insert(varId);
        ssaComps[std::make_pair(varId, idxId)] = &*ii;
      }
      else {
        if (ssaCompVars.find(varId) != ssaCompVars.end()) {
          nonSsaVars.insert(varId);
          continue;
        }
        ssaVars[varId] = &*ii;
      }
    }
  }
}

bool SSAMemPass::SSAMemProcess(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      if (ii->opcode() != SpvOpLoad)
        continue;
      const uint32_t ptrId = ii->GetInOperand(SPV_STORE_POINTER_ID).words[0];
      const ir::Instruction* ptrInst =
        def_use_mgr_->id_to_defs().find(ptrId)->second;
      const uint32_t varId = ptrInst->opcode() == SpvOpAccessChain ?
        ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0] :
        ptrId;
      if (nonSsaVars.find(varId) != nonSsaVars.end())
        continue;
      uint32_t replId;
      if (ptrInst->opcode() == SpvOpAccessChain) {
        // process component load
        const uint32_t idxId =
          ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
        const auto csi = ssaComps.find(std::make_pair(varId, idxId));
        if (csi != ssaComps.end())
          replId = csi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
        else {
          // See if the whole variable stored with a load of an SSA var.
          // If so, look for a component store into the load variable
          // and use the value Id that was stored.
          const auto vsi = ssaVars.find(varId);
          if (vsi == ssaVars.end())
            continue;
          const uint32_t valId =
              vsi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          const ir::Instruction* valInst =
              def_use_mgr_->id_to_defs().find(valId)->second;
          if (valInst->opcode() != SpvOpLoad)
            continue;
          const uint32_t loadVarId =
              valInst->GetInOperand(SPV_LOAD_PTR_ID).words[0];
          if (nonSsaVars.find(loadVarId) != nonSsaVars.end())
            continue;
          const auto lvcsi = ssaComps.find(std::make_pair(loadVarId, idxId));
          if (lvcsi == ssaComps.end())
            continue;
          replId = lvcsi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
        }
      }
      else {
        // process whole variable load
        const auto vsi = ssaVars.find(varId);
        // if variable is not defined with whole variable store,
        // skip this load
        if (vsi == ssaVars.end())
          continue;
        replId = vsi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
      }
      const uint32_t loadId = ii->result_id();
      const bool replaced = def_use_mgr_->ReplaceAllUsesWith(loadId, replId);
      if (replaced)
        modified = true;
    }
  }
  return modified;
}

bool SSAMemPass::SSAMem(ir::Function* func) {
    SSAMemAnalyze(func);
    bool modified = SSAMemProcess(func);
    return modified;
}

void SSAMemPass::Initialize(ir::Module* module) {
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

  // Initialize next unused Id
  nextId_ = 0;
  for (const auto& id_def : def_use_mgr_->id_to_defs()) {
    nextId_ = std::max(nextId_, id_def.first);
  }
  nextId_++;

  module_ = module;

  // Initialize function and block maps
  id2function.clear();
  for (auto& fn : *module_) {
    id2function[fn.GetResultId()] = &fn;
  }

  // Initialize Target Type Caches
  seenTargetTypes.clear();
  seenNonTargetTypes.clear();
};

Pass::Status SSAMemPass::ProcessImpl() {
  // do exhaustive inlining on each entry point function in module
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function[e.GetOperand(SPV_ENTRY_POINT_FUNCTION_ID).words[0]];
    modified = modified || SSAMem(fn);
  }

  finalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

SSAMemPass::SSAMemPass()
    : nextId_(0), module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status SSAMemPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

