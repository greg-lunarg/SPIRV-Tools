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
static const int SPV_STORE_PTR_ID = 0;
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
  if (typeInst->opcode() != SpvOpTypeStruct &&
      typeInst->opcode() != SpvOpTypeArray)
    return false;
  int nonMathComp = 0;
  typeInst->ForEachInId([&nonMathComp,this](uint32_t* tid) {
    ir::Instruction* compTypeInst =
        def_use_mgr_->id_to_defs().find(*tid)->second;
    // Ignore length operand in Array type
    if (compTypeInst->opcode() == SpvOpConstant) return;
    if (!isMathType(compTypeInst)) nonMathComp++;
  });
  return nonMathComp == 0;
}

ir::Instruction* SSAMemPass::GetPtr(ir::Instruction* ip, uint32_t& varId) {
  const uint32_t ptrId = ip->opcode() == SpvOpStore ?
      ip->GetInOperand(SPV_STORE_PTR_ID).words[0] :
      ip->GetInOperand(SPV_LOAD_PTR_ID).words[0];
  ir::Instruction* ptrInst =
    def_use_mgr_->id_to_defs().find(ptrId)->second;
  varId = ptrInst->opcode() == SpvOpAccessChain ?
    ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0] :
    ptrId;
  return ptrInst;
}

bool SSAMemPass::isTargetPtr(uint32_t varId) {
  const ir::Instruction* varInst =
    def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  if (seenNonTargetTypes.find(varTypeId) != seenNonTargetTypes.end())
    return false;
  if (seenTargetTypes.find(varTypeId) == seenTargetTypes.end()) {
    const ir::Instruction* varTypeInst =
      def_use_mgr_->id_to_defs().find(varTypeId)->second;
    if (varTypeInst->GetInOperand(SPV_TYPE_PTR_STORAGE_CLASS).words[0] !=
      SpvStorageClassFunction) {
      seenNonTargetTypes.insert(varTypeId);
      return false;
    }
    const uint32_t varPteTypeId =
      varTypeInst->GetInOperand(SPV_TYPE_PTR_TYPE_ID).words[0];
    ir::Instruction* varPteTypeInst =
      def_use_mgr_->id_to_defs().find(varPteTypeId)->second;
    if (!isTargetType(varPteTypeInst)) {
      seenNonTargetTypes.insert(varTypeId);
      return false;
    }
    seenTargetTypes.insert(varTypeId);
  }
  return true;
}

void SSAMemPass::SSAMemAnalyze(ir::Function* func) {
  ssaVars.clear();
  ssaComps.clear();
  ssaCompVars.clear();
  nonSsaVars.clear();
  storeIdx.clear();
  uint32_t instIdx = 0;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++, instIdx++) {
      if (ii->opcode() != SpvOpStore)
        continue;

      // Verify store variable is target type
      uint32_t varId;
      ir::Instruction* ptrInst = GetPtr(&*ii, varId);
      if (!isTargetPtr(varId)) {
        nonSsaVars.insert(varId);
        continue;
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
        storeIdx[&*ii] = instIdx;
      }
      else {
        if (ssaCompVars.find(varId) != ssaCompVars.end()) {
          nonSsaVars.insert(varId);
          continue;
        }
        ssaVars[varId] = &*ii;
        storeIdx[&*ii] = instIdx;
      }
    }
  }
}

void SSAMemPass::ReplaceAndDeleteLoad(ir::Instruction* loadInst,
                                      uint32_t replId,
                                      ir::Instruction* ptrInst) {
  const uint32_t loadId = loadInst->result_id();
  (void) def_use_mgr_->ReplaceAllUsesWith(loadId, replId);
  // remove load instruction
  def_use_mgr_->KillInst(loadInst);
  // if access chain, see if it can be removed as well
  if (ptrInst->opcode() == SpvOpAccessChain) {
    const uint32_t ptrId = ptrInst->result_id();
    analysis::UseList* uses = def_use_mgr_->GetUses(ptrId);
    if (uses == nullptr)
      def_use_mgr_->KillDef(ptrId);
  }
}

bool SSAMemPass::SSAMemProcess(ir::Function* func) {
  bool modified = false;
  uint32_t instIdx = 0;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++, instIdx++) {
      if (ii->opcode() != SpvOpLoad)
        continue;
      const uint32_t ptrId = ii->GetInOperand(SPV_LOAD_PTR_ID).words[0];
      ir::Instruction* ptrInst =
          def_use_mgr_->id_to_defs().find(ptrId)->second;
      const uint32_t varId = ptrInst->opcode() == SpvOpAccessChain ?
          ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0] :
          ptrId;
      const ir::Instruction* varInst =
          def_use_mgr_->id_to_defs().find(varId)->second;
      assert(varInst->opcode() == SpvOpVariable);
      if (nonSsaVars.find(varId) != nonSsaVars.end())
        continue;
      uint32_t replId;
      uint32_t sIdx;
      if (ptrInst->opcode() == SpvOpAccessChain) {
        if (ptrInst->NumInOperands() != 2)
          continue;
        // process component load
        const uint32_t idxId =
            ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
        const auto csi = ssaComps.find(std::make_pair(varId, idxId));
        if (csi != ssaComps.end()) {
          replId = csi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          sIdx = storeIdx[csi->second];
        }
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
          sIdx = storeIdx[lvcsi->second];
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
        sIdx = storeIdx[vsi->second];
      }
      // store must dominate load
      if (instIdx < sIdx)
        continue;
      // replace all instances of the load's id with the SSA value's id
      ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
      modified = true;
    }
  }
  return modified;
}

bool SSAMemPass::isLiveVar(uint32_t varId) {
  // non-function scope vars are live
  const ir::Instruction* varInst =
      def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
      def_use_mgr_->id_to_defs().find(varTypeId)->second;
  if (varTypeInst->GetInOperand(SPV_TYPE_PTR_STORAGE_CLASS).words[0] !=
      SpvStorageClassFunction)
    return true;
  // test if variable is loaded from
  analysis::UseList* uses = def_use_mgr_->GetUses(varId);
  if (uses->size() <= 1)
    return false;
  for (auto u : *uses) {
    if (u.inst->opcode() == SpvOpAccessChain) {
      uint32_t cid = u.inst->result_id();
      analysis::UseList* cuses = def_use_mgr_->GetUses(cid);
      if (cuses == nullptr)
        continue;
      for (auto cu : *cuses)
        if (cu.inst->opcode() == SpvOpLoad)
          return true;
    }
    else if (u.inst->opcode() == SpvOpLoad)
      return true;
  }
  return false;
}

bool SSAMemPass::isLiveStore(ir::Instruction* storeInst) {
  // get store's variable
  uint32_t varId;
  (void) GetPtr(storeInst, varId);
  return isLiveVar(varId);
}

void SSAMemPass::DCEInst(ir::Instruction* inst) {
  std::queue<ir::Instruction*> deadInsts;
  deadInsts.push(inst);
  while (!deadInsts.empty()) {
    ir::Instruction* di = deadInsts.front();
    std::queue<uint32_t> ids;
    di->ForEachInId([&ids](uint32_t* iid) {
      ids.push(*iid);
    });
    uint32_t varId = 0;
    if (di->opcode() == SpvOpLoad)
      (void) GetPtr(di, varId);
    def_use_mgr_->KillInst(di);
    while (!ids.empty()) {
      uint32_t id = ids.front();
      analysis::UseList* uses = def_use_mgr_->GetUses(id);
      if (uses == nullptr)
        deadInsts.push(def_use_mgr_->GetDef(id));
      ids.pop();
    }
    // if a load was deleted and it was the variable's
    // last load, add all its stores to dead queue
    if (varId != 0 && !isLiveVar(varId)) {
      analysis::UseList* uses = def_use_mgr_->GetUses(varId);
      if (uses != nullptr) {
        for (auto u : *uses) {
          if (u.inst->opcode() == SpvOpAccessChain) {
            uint32_t cid = u.inst->result_id();
            analysis::UseList* cuses = def_use_mgr_->GetUses(cid);
            if (cuses != nullptr) {
              for (auto cu : *cuses)
                if (cu.inst->opcode() == SpvOpStore)
                  deadInsts.push(cu.inst);
            }
          }
          else if (u.inst->opcode() == SpvOpStore)
            deadInsts.push(u.inst);
        }
      }
    }
    deadInsts.pop();
  }
}

bool SSAMemPass::SSAMemDCE() {
  bool modified = false;
  for (auto v : ssaVars) {
    // check that it hasn't already been DCE'd
    if (v.second->opcode() != SpvOpStore)
      continue;
    if (nonSsaVars.find(v.first) != nonSsaVars.end())
      continue;
    if (!isLiveStore(v.second)) {
      DCEInst(v.second);
      modified = true;
    }
  }
  for (auto c : ssaComps) {
    // check that it hasn't already been DCE'd
    if (c.second->opcode() != SpvOpStore)
      continue;
    if (nonSsaVars.find(c.first.first) != nonSsaVars.end())
      continue;
    if (!isLiveStore(c.second)) {
      DCEInst(c.second);
      modified = true;
    }
  }
  return modified;
}

void SSAMemPass::SBEraseComps(uint32_t varId) {
  for (auto ci = sbCompStores.begin(); ci != sbCompStores.end();)
    if (ci->first.first == varId)
      ci = sbCompStores.erase(ci);
    else
      ci++;
  for (auto ci = sbPinnedComps.begin(); ci != sbPinnedComps.end();)
    if (ci->first == varId)
      ci = sbPinnedComps.erase(ci);
    else
      ci++;
}

bool SSAMemPass::SSAMemSingleBlock(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    sbVarStores.clear();
    sbVarLoads.clear();
    sbCompStores.clear();
    sbPinnedVars.clear();
    sbPinnedComps.clear();
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        // Verify store variable is target type
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (!isTargetPtr(varId))
          continue;
        // Register the store
        if (ptrInst->opcode() == SpvOpVariable) {
          // if not pinned, look for WAW
          if (sbPinnedVars.find(varId) == sbPinnedVars.end()) {
            auto si = sbVarStores.find(varId);
            if (si != sbVarStores.end()) {
              def_use_mgr_->KillInst(si->second);
            }
          }
          sbVarStores[varId] = &*ii;
          SBEraseComps(varId);
        }
        else {
          assert(ptrInst->opcode() == SpvOpAccessChain);
          const uint32_t idxId =
              ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
          if (ptrInst->NumInOperands() == 2) {
            // if not pinned, look for WAW
            if (sbPinnedComps.find(std::make_pair(varId, idxId)) == sbPinnedComps.end()) {
              auto si = sbCompStores.find(std::make_pair(varId, idxId));
              if (si != sbCompStores.end()) {
                DCEInst(si->second);
              }
            }
            sbCompStores[std::make_pair(varId, idxId)] = &*ii;
          }
          else
            sbCompStores.erase(std::make_pair(varId, idxId));
          sbPinnedComps.erase(std::make_pair(varId, idxId));
          sbVarStores.erase(varId);
        }
        sbPinnedVars.erase(varId);
        sbVarLoads.erase(varId);
      } break;
      case SpvOpLoad: {
        // Verify store variable is target type
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (!isTargetPtr(varId))
          continue;
        // Look for previous store or load
        uint32_t replId = 0;
        if (ptrInst->opcode() == SpvOpVariable) {
          auto si = sbVarStores.find(varId);
          if (si != sbVarStores.end()) {
            replId = si->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          }
          else {
            auto li = sbVarLoads.find(varId);
            if (li != sbVarLoads.end()) {
              replId = li->second->result_id();
            }
          }
        }
        else if (ptrInst->NumInOperands() == 2) {
          assert(ptrInst->opcode() == SpvOpAccessChain);
          const uint32_t idxId =
            ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
          auto ci = sbCompStores.find(std::make_pair(varId, idxId));
          if (ci != sbCompStores.end()) {
            replId = ci->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          }
          else {
            // if live store to whole variable of load, look for component
            // store to the loaded variable
            auto si = sbVarStores.find(varId);
            if (si != sbVarStores.end()) {
              const uint32_t valId =
                si->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
              ir::Instruction* valInst =
                def_use_mgr_->id_to_defs().find(valId)->second;
              if (valInst->opcode() == SpvOpLoad) {
                uint32_t loadVarId =
                  valInst->GetInOperand(SPV_LOAD_PTR_ID).words[0];
                auto lvi = sbCompStores.find(std::make_pair(loadVarId, idxId));
                if (lvi != sbCompStores.end()) {
                  replId = lvi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
                }
              }
            }
          }
        }
        if (replId != 0) {
          // replace load's result id and delete load
          ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
          modified = true;
        }
        else {
          if (ptrInst->opcode() == SpvOpVariable) {
            sbVarLoads[varId] = &*ii;  // register load
            sbPinnedVars.insert(varId);
            for (auto ci = sbCompStores.begin(); ci != sbCompStores.end(); ci++)
              if (ci->first.first == varId)
                sbPinnedComps.insert(std::make_pair(varId, ci->first.second));
          }
          else {
            const uint32_t idxId =
              ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
            if (sbCompStores.find(std::make_pair(varId, idxId)) != sbCompStores.end())
              sbPinnedComps.insert(std::make_pair(varId, idxId));
            else
              sbPinnedVars.insert(varId);
          }
        }
      } break;
      default:
        break;
      }
    }
    // Go back and delete useless stores in block
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      if (ii->opcode() != SpvOpStore)
        continue;
      if (isLiveStore(&*ii))
        continue;
      DCEInst(&*ii);
    }
  }
  return modified;
}

bool SSAMemPass::SSAMemDCEFunc(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        (void)GetPtr(&*ii, varId);
        if (isLiveVar(varId))
          break;
        DCEInst(&*ii);
        modified = true;
      } break;
      default: {
        uint32_t rId = ii->result_id();
        if (rId == 0)
          break;
        analysis::UseList* uses = def_use_mgr_->GetUses(rId);
        if (uses != nullptr)
          break;
        DCEInst(&*ii);
        modified = true;
      } break;
      }
    }
  }
  return modified;
}

bool SSAMemPass::SSAMem(ir::Function* func) {
    bool modified = false;
    modified |= SSAMemSingleBlock(func);
    SSAMemAnalyze(func);
    modified |= SSAMemProcess(func);
    modified |= SSAMemDCE();
    modified |= SSAMemDCEFunc(func);
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

