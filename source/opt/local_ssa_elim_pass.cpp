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

#include "iterator.h"
#include "local_single_block_elim_pass.h"

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvStorePtrId = 0;
static const int kSpvStoreValId = 1;
static const int kSpvLoadPtrId = 0;
static const int kSpvAccessChainPtrId = 0;
static const int kSpvTypePointerStorageClass = 0;
static const int kSpvTypePointerTypeId = 1;

static const int kSpvFunctionCallFunctionId = 0;
static const int kSpvDecorationTargetId = 0;
static const int kSpvDecorationDecoration = 1;
static const int kSpvDecorationLinkageType = 3;

static const int kSpvConstantValue = 0;
static const int kSpvExtractCompositeId = 0;
static const int kSpvExtractIdx0 = 1;
static const int kSpvInsertObjectId = 0;
static const int kSpvInsertCompositeId = 1;
static const int kSpvBranchCondConditionalId = 0;
static const int kSpvBranchCondTrueLabId = 1;
static const int kSpvBranchCondFalseLabId = 2;
static const int kSpvSelectionMergeMergeBlockId = 0;
static const int kSpvPhiVal0Id = 0;
static const int kSpvPhiLab0Id = 1;
static const int kSpvPhiVal1Id = 2;
static const int kSpvLoopMergeMergeBlockId = 0;

namespace spvtools {
namespace opt {

bool LocalSSAElimPass::IsNonPtrAccessChain(
    const SpvOp opcode) const {
  return opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain;
}

bool LocalSSAElimPass::IsMathType(
    const ir::Instruction* typeInst) const {
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

bool LocalSSAElimPass::IsTargetType(
    const ir::Instruction* typeInst) const {
  if (IsMathType(typeInst))
    return true;
  if (typeInst->opcode() == SpvOpTypeArray)
    return IsMathType(def_use_mgr_->GetDef(typeInst->GetSingleWordOperand(1)));
  if (typeInst->opcode() != SpvOpTypeStruct)
    return false;
  // All struct members must be math type
  int nonMathComp = 0;
  typeInst->ForEachInId([&nonMathComp,this](const uint32_t* tid) {
    ir::Instruction* compTypeInst = def_use_mgr_->GetDef(*tid);
    if (!IsMathType(compTypeInst)) ++nonMathComp;
  });
  return nonMathComp == 0;
}

ir::Instruction* LocalSSAElimPass::GetPtr(
      ir::Instruction* ip, uint32_t* varId) {
  *varId = ip->GetSingleWordInOperand(
      ip->opcode() == SpvOpStore ?  kSpvStorePtrId : kSpvLoadPtrId);
  ir::Instruction* ptrInst = def_use_mgr_->GetDef(*varId);
  ir::Instruction* varInst = ptrInst;
  while (IsNonPtrAccessChain(varInst->opcode())) {
    *varId = varInst->GetSingleWordInOperand(kSpvAccessChainPtrId);
    varInst = def_use_mgr_->GetDef(*varId);
  }
  return ptrInst;
}

bool LocalSSAElimPass::IsTargetVar(uint32_t varId) {
  if (seen_non_target_vars_.find(varId) != seen_non_target_vars_.end())
    return false;
  if (seen_target_vars_.find(varId) != seen_target_vars_.end())
    return true;
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = def_use_mgr_->GetDef(varTypeId);
  if (varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) !=
    SpvStorageClassFunction) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  const uint32_t varPteTypeId =
    varTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeId);
  ir::Instruction* varPteTypeInst = def_use_mgr_->GetDef(varPteTypeId);
  if (!IsTargetType(varPteTypeInst)) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  seen_target_vars_.insert(varId);
  return true;
}

bool LocalSSAElimPass::HasLoads(uint32_t ptrId) const {
  analysis::UseList* uses = def_use_mgr_->GetUses(ptrId);
  if (uses == nullptr)
    return false;
  for (auto u : *uses) {
    SpvOp op = u.inst->opcode();
    if (IsNonPtrAccessChain(op)) {
      if (HasLoads(u.inst->result_id()))
        return true;
    }
    else {
      // Conservatively assume that calls will do a load
      // TODO(): Improve analysis around function calls
      if (op == SpvOpLoad || op == SpvOpFunctionCall)
        return true;
    }
  }
  return false;
}

bool LocalSSAElimPass::IsLiveVar(uint32_t varId) const {
  // non-function scope vars are live
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = def_use_mgr_->GetDef(varTypeId);
  if (varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) !=
      SpvStorageClassFunction)
    return true;
  // test if variable is loaded from
  return HasLoads(varId);
}

void LocalSSAElimPass::AddStores(
    uint32_t ptr_id, std::queue<ir::Instruction*>* insts) {
  analysis::UseList* uses = def_use_mgr_->GetUses(ptr_id);
  if (uses != nullptr) {
    for (auto u : *uses) {
      if (IsNonPtrAccessChain(u.inst->opcode()))
        AddStores(u.inst->result_id(), insts);
      else if (u.inst->opcode() == SpvOpStore)
        insts->push(u.inst);
    }
  }
}

void LocalSSAElimPass::DCEInst(ir::Instruction* inst) {
  std::queue<ir::Instruction*> deadInsts;
  deadInsts.push(inst);
  while (!deadInsts.empty()) {
    ir::Instruction* di = deadInsts.front();
    // Don't delete labels
    if (di->opcode() == SpvOpLabel) {
      deadInsts.pop();
      continue;
    }
    // Remember operands
    std::vector<uint32_t> ids;
    di->ForEachInId([&ids](uint32_t* iid) {
      ids.push_back(*iid);
    });
    uint32_t varId = 0;
    // Remember variable if dead load
    if (di->opcode() == SpvOpLoad)
      (void) GetPtr(di, &varId);
    def_use_mgr_->KillInst(di);
    // For all operands with no remaining uses, add their instruction
    // to the dead instruction queue.
    for (auto id : ids) {
      analysis::UseList* uses = def_use_mgr_->GetUses(id);
      if (uses == nullptr)
        deadInsts.push(def_use_mgr_->GetDef(id));
    }
    // if a load was deleted and it was the variable's
    // last load, add all its stores to dead queue
    if (varId != 0 && !IsLiveVar(varId)) 
      AddStores(varId, &deadInsts);
    deadInsts.pop();
  }
}

bool LocalSingleStoreElimPass::HasOnlySupportedRefs(uint32_t varId) {
  if (supported_ref_vars_.find(varId) != supported_ref_vars_.end())
    return true;
  analysis::UseList* uses = def_use_mgr_->GetUses(varId);
  assert(uses != nullptr);
  for (auto u : *uses) {
    SpvOp op = u.inst->opcode();
    if (op != SpvOpStore && op != SpvOpLoad && op != SpvOpName)
      return false;
  }
  supported_ref_vars_.insert(varId);
  return true;
}

void LocalSSAElim::InitSSARewrite(ir::Function& func) {
  // Init predecessors
  label2preds_.clear();
  for (auto& blk : func) {
    uint32_t blkId = blk.GetLabelId();
    blk.ForEachSucc([&blkId, this](uint32_t sbid) {
      label2preds_[sbid].push_back(blkId);
    });
  }
  // Remove variables with non-load/store refs from target variable set
  for (auto& blk : func) {
    for (auto& inst : blk) {
      switch (ii->opcode()) {
      case SpvOpStore:
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsTargetVar(varId))
          break;
        if (HasOnlySupportedRefs(varId))
          break;
        seen_non_target_vars_.insert(varId);
        seen_target_vars_.erase(varId);
      } break;
      default:
        break;
      }
    }
  }
}

uint32_t LocalSSAElimPass::MergeBlockIdIfAny(const ir::BasicBlock& blk) {
  auto merge_ii = blk.cend();
  --merge_ii;
  uint32_t mbid = 0;
  if (merge_ii != blk.cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == SpvOpLoopMerge)
      mbid = merge_ii->GetSingleWordOperand(kSpvLoopMergeMergeBlockId);
    else if (merge_ii->opcode() == SpvOpSelectionMerge)
      mbid = merge_ii->GetSingleWordOperand(kSpvSelectionMergeMergeBlockId);
  }
  return mbid;
}

void LocalSSAElimPass::ComputeStructuredSuccessors(ir::Function* func) {
  // If header, make merge block first successor.
  for (auto& blk : *func) {
    uint32_t mbid = MergeBlockIdIfAny(blk);
    if (mbid != 0)
      block2structured_succs_[&blk].push_back(id2block_[mbid]);
    // add true successors
    blk.ForEachSuccessorLabel([&blk, this](uint32_t sbid) {
      block2structured_succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}

LocalSSAElimPass::GetBlocksFunction
LocalSSAElimPass::StructuredSuccessorsFunction() {
  return [this](const ir::BasicBlock* block) {
    return &(block2structured_succs_[block]);
  };
}

void LocalSSAElim::ComputeStructuredOrder(ir::Function& func,
    std::list<ir::BasicBlock*>* structuredOrder) {
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  structuredOrder->clear();
  spvtools::CFA<ir::BasicBlock>::DepthFirstTraversal(
    &*func->begin(), StructuredSuccessorsFunction(), ignore_block,
    [&](cbb_ptr b) { structuredOrder->push_front(b); }, ignore_edge);
}

void LocalSSAElimPass::SSABlockInitSinglePred(ir::BasicBlock* block_ptr) {
  uint32_t label = block_ptr->GetLabelId();
  uint32_t predLabel = label2preds_[label].front();
  assert(visitedBlocks.find(predLabel) != visitedBlocks.end());
  label2ssa_map_[label] = label2ssa_map_[predLabel];
}

bool LocalSSAElimPass::IsLiveAfter(uint32_t var_id, uint32_t label) {
  // For now, return very conservative result: true. This will result in
  // correct, but possibly usused, phi code to be generated. A subsequent
  // DCE pass should eliminate this code.
  // TODO(): Return more accurate information
  (void) var_id;
  (void) label;
  return true;
}

uint32_t LocalSSAElimPass::Type2Undef(uint32_t type_id) {
  const auto uitr = type2undefs_.find(type_id);
  if (uitr != type2undefs_.end())
    return uitr->second;
  uint32_t undefId = TakeNextId();
  std::unique_ptr<ir::Instruction> undef_inst(
    new ir::Instruction(SpvOpUndef, type_id, undefId, {}));
  def_use_mgr_->AnalyzeInstDefUse(&*undef_inst);
  module_->AddGlobalValue(std::move(undef_inst));
  type2undefs_[type_id] = undefId;
  return undefId;
}

void LocalSSAElimPass::SSABlockInitLoopHeader(
    ir::UptrVectorIterator<ir::BasicBlock> block_itr) {
  uint32_t label = block_itr->GetLabelId();
  // Determine backedge label.
  uint32_t backLabel = 0;
  for (uint32_t predLabel : label2preds_[label])
    if (visitedBlocks.find(predLabel) == visitedBlocks.end()) {
      assert(backLabel == 0);
      backLabel = predLabel;
      break;
    }
  assert(backLabel != 0);
  // Determine merge block.
  auto mergeInst = block_itr->begin();
  uint32_t mergeLabel = mergeInst->GetSingleWordInOperand(
      kSpvLoopMergeMergeBlockId);
  // Collect all live variables and a default value for each across all
  // non-backedge predecesors.
  std::unordered_map<uint32_t, uint32_t> liveVars;
  for (uint32_t predLabel : label2preds_[label]) {
    for (auto var_val : label2ssa_map_[predLabel]) {
      uint32_t varId = var_val.first;
      liveVars[varId] = var_val.second;
    }
  }
  // Add all stored variables in loop. Set their default value id to zero.
  for (auto bi = block_itr; bi->GetLabelId() != mergeLabel; ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpStore)
        continue;
      uint32_t varId;
      (void) GetPtr(&*ii, &varId);
      if (!IsTargetVar(varId))
        break;
      liveVars[varId] = 0;
    }
  }
  // Insert phi for all live variables that require them. All variables
  // defined in loop require a phi. Otherwise all variables
  // with differing predecessor values require a phi.
  auto insertItr = block_itr->begin();
  for (auto var_val : liveVars) {
    uint32_t varId = var_val.first;
    if (!IsLiveAfter(varId, label))
      continue;
    uint32_t val0Id = var_val.second;
    bool needsPhi = false;
    if (val0Id != 0) {
      for (uint32_t predLabel : label2preds_[label]) {
        // Skip back edge predecessor.
        if (predLabel == backLabel)
          continue;
        const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
        // Missing values do not cause difference
        if (var_val_itr == label2ssa_map_[predLabel].end())
          continue;
        if (var_val_itr->second != val0Id) {
          needsPhi = true;
          break;
        }
      }
    }
    else {
      needsPhi = true;
    }
    // If val is the same for all predecessors, enter it in map
    if (!needsPhi) {
      label2ssa_map_[label].insert(var_val);
      continue;
    }
    // Val differs across predecessors. Add phi op to block and 
    // add its result id to the map. For live back edge predecessor,
    // use the variable id. We will patch this after visiting back
    // edge predecessor. For predessors that do not define a value,
    // use undef.
    std::vector<ir::Operand> phi_in_operands;
    uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
    for (uint32_t predLabel : label2preds_[label]) {
      uint32_t valId;
      if (predLabel == backLabel) {
        if (val0Id == 0)
                                                              896,1         58%
          valId = varId;
        else
          valId = Type2Undef(typeId);
      }
      else {
        const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
        if (var_val_itr == label2ssa_map_[predLabel].end())
          valId = Type2Undef(typeId);
        else
          valId = var_val_itr->second;
      }
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{valId}));
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{predLabel}));
    }
    uint32_t phiId = TakeNextId();
    std::unique_ptr<ir::Instruction> newPhi(
      new ir::Instruction(SpvOpPhi, typeId, phiId, phi_in_operands));
    // Only analyze the phi define now; analyze the phi uses after the
    // phi backedge predecessor value is patched.
    def_use_mgr_->AnalyzeInstDef(&*newPhi);
    insertItr = insertItr.InsertBefore(std::move(newPhi));
    ++insertItr;
    label2ssa_map_[label].insert({ varId, phiId });
  }
}

void LocalSSAElimPass::SSABlockInitSelectMerge(ir::BasicBlock* block_ptr) {
  uint32_t label = block_ptr->GetLabelId();
  // Collect all live variables and a default value for each across all
  // predecesors.
  std::unordered_map<uint32_t, uint32_t> liveVars;
  for (uint32_t predLabel : label2preds_[label]) {
    assert(visitedBlocks.find(predLabel) != visitedBlocks.end());
    for (auto var_val : label2ssa_map_[predLabel]) {
      uint32_t varId = var_val.first;
      liveVars[varId] = var_val.second;
    }
  }
  // For each live variable, look for a difference in values across
  // predecessors that would require a phi and insert one.
  auto insertItr = block_ptr->begin();
  for (auto var_val : liveVars) {
    uint32_t varId = var_val.first;
    if (!IsLiveAfter(varId, label))
      continue;
    uint32_t val0Id = var_val.second;
    bool differs = false;
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
      // Missing values cause a difference
      if (var_val_itr == label2ssa_map_[predLabel].end()) {
        differs = true;
        break;
      }
      if (var_val_itr->second != val0Id) {
        differs = true;
        break;
      }
    }
    // If val is the same for all predecessors, enter it in SSA map
    if (!differs) {
      label2ssa_map_[label].insert(var_val);
      continue;
    }
    // Val differs across predecessors. Add phi op to block and 
    // add its result id to the map
    std::vector<ir::Operand> phi_in_operands;
    uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
      // If variable not defined on this path, use undef
      uint32_t valId = (var_val_itr != label2ssa_map_[predLabel].end()) ?
          var_val_itr->second : Type2Undef(typeId);
      phi_in_operands.push_back(
          ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{valId}));
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{predLabel}));
    }
    uint32_t phiId = TakeNextId();
    std::unique_ptr<ir::Instruction> newPhi(
      new ir::Instruction(SpvOpPhi, typeId, phiId, phi_in_operands));
    def_use_mgr_->AnalyzeInstDefUse(&*newPhi);
    insertItr = insertItr.InsertBefore(std::move(newPhi));
    ++insertItr;
    label2ssa_map_[label].insert({varId, phiId});
  }
}

bool LocalSSAElimPass::IsLoopHeader(ir::BasicBlock* block_ptr) {
  auto iItr = block_ptr->end();
  --iItr;
  if (iItr == block_ptr->begin())
    return false;
  --iItr;
  return iItr->opcode() == SpvOpLoopMerge;
}

void LocalSSAElimPass::SSABlockInit(
    std::list<ir::BasicBlock*>::iterator block_itr) {
  size_t numPreds = label2preds_[block_itr->GetLabelId()].size();
  if (numPreds == 0)
    return;
  if (numPreds == 1)
    SSABlockInitSinglePred(&*block_itr);
  else if (IsLoopHeader(&*block_itr))
    SSABlockInitLoopHeader(block_itr);
  else
    SSABlockInitSelectMerge(&*block_itr);
}

void LocalSSAElimPass::PatchPhis(uint32_t header_id, uint32_t back_id) {
  ir::BasicBlock* header = label2block_[header_id];
  auto phiItr = header->begin();
  for (; phiItr->opcode() == SpvOpPhi; ++phiItr) {
    uint32_t cnt = 0;
    uint32_t idx;
    phiItr->ForEachInId([&cnt,&back_id,&idx](uint32_t* iid) {
      if (cnt % 2 == 1 && *iid == back_id) idx = cnt - 1;
      ++cnt;
    });
    // Only patch operands that are in the backedge predecessor map
    uint32_t varId = phiItr->GetSingleWordInOperand(idx);
    const auto valItr = label2ssa_map_[back_id].find(varId);
    if (valItr != label2ssa_map_[back_id].end()) {
      phiItr->SetInOperand(idx, { valItr->second });
      // Analyze uses now that they are complete
      def_use_mgr_->AnalyzeInstUse(&*phiItr);
    }
  }
}

bool LocalSSAElimPass::LocalSSAElim(ir::Function* func) {
  // Assumes all control flow structured.
  // TODO: Do SSA rewrite for non-structured control flow
  if (!module_->HasCapability(SpvCapabilityShader))
    return false;
  InitSSARewrite(*func);
  std::list<ir::BasicBlock*> structuredOrder;
  ComputeStructuredOrder(func, &structuredOrder);
  bool modified = false;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    // Initialize this block's SSA map using predecessor's maps.
    SSABlockInit(bi);
    uint32_t label = bi->GetLabelId();
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsTargetVar(varId))
          break;
        assert(ptrInst->opcode() != SpvOpAccessChain);
        uint32_t valId = ii->GetInOperand(kSpvStoreValId).words[0];
        // Register new stored value for the variable
        label2ssa_map_[label][varId] = valId;
      } break;
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsTargetVar(varId))
          break;
        assert(ptrInst->opcode() != SpvOpAccessChain);
        // If variable is not defined, use undef
        uint32_t replId = 0;
        const auto ssaItr = label2ssa_map_.find(label);
        if (ssaItr != label2ssa_map_.end()) {
          const auto valItr = ssaItr->second.find(varId);
          if (valItr != ssaItr->second.end())
            replId = valItr->second;
        }
        if (replId == 0) {
          uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
          replId = Type2Undef(typeId);
        }
        // Replace load's id with the last stored value id
        // and delete load.
        const uint32_t loadId = ii->result_id();
        (void)def_use_mgr_->ReplaceAllUsesWith(loadId, replId);
        def_use_mgr_->KillInst(&*ii);
        modified = true;
      } break;
      default: {
      } break;
      }
    }
    visitedBlocks.insert(label);
    // Look for successor backedge and patch phis in loop header
    // if found.
    uint32_t header = 0;
    bi->ForEachSucc([&header,this](uint32_t succ) {
      if (visitedBlocks.find(succ) == visitedBlocks.end()) return;
      assert(header == 0);
      header = succ;
    });
    if (header != 0)
      PatchPhis(header, label);
  }
  // Remove all target variable stores.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpStore)
        continue;
      uint32_t varId;
      (void) GetPtr(&*ii, &varId);
      if (!IsTargetVar(varId))
        break;
      assert(!HasLoads(varId));
      DCEInst(&*ii);
      modified = true;
    }
  }
  return modified;
}

void LocalSSAElimPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  for (auto& fn : *module_) 
    id2function_[fn.result_id()] = &fn;

  // Initialize Target Type Caches
  seen_target_vars_.clear();
  seen_non_target_vars_.clear();

  // Initialize set of variables only referenced by supported operations
  supported_ref_vars_.clear();

  // TODO(): Reuse def/use from previous passes
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));

  // Start new ids with next availablein module
  next_id_ = module_->id_bound();

};

Pass::Status LocalSSAElimPass::ProcessImpl() {
  // Assumes logical addressing only
  if (module_->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;
  bool modified = false;
  // Call Mem2Reg on all remaining functions.
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || LocalSSAElim(fn);
  }
  FinalizeNextId(module_);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LocalSSAElimPass::LocalSSAElimPass()
    : module_(nullptr), def_use_mgr_(nullptr), next_id_(0) {}

Pass::Status LocalSSAElimPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

