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

#include "aggressive_dce_pass.h"

#include "cfa.h"
#include "iterator.h"

static const int kSpvEntryPointFunctionId = 1;
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

uint32_t AggressiveDCEPass::MergeBlockIdIfAny(
    const ir::BasicBlock& blk) const {
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

void AggressiveDCEPass::ComputeStructuredSuccessors(ir::Function* func) {
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

AggressiveDCEPass::GetBlocksFunction
AggressiveDCEPass::StructuredSuccessorsFunction() {
  return [this](const ir::BasicBlock* block) {
    return &(block2structured_succs_[block]);
  };
}

void AggressiveDCEPass::ComputeStructuredOrder(
    ir::Function* func, std::list<ir::BasicBlock*>* order) {
  // Compute structured successors and do DFS
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  spvtools::CFA<ir::BasicBlock>::DepthFirstTraversal(
    &*func->begin(), StructuredSuccessorsFunction(), ignore_block,
    [&](cbb_ptr b) { order->push_front(const_cast<ir::BasicBlock*>(b)); }, ignore_edge);
}

void AggressiveDCEPass::GetConstCondition(
    uint32_t condId, bool* condVal, bool* condIsConst) {
  ir::Instruction* cInst = def_use_mgr_->GetDef(condId);
  switch (cInst->opcode()) {
  case SpvOpConstantFalse: {
    *condVal = false;
    *condIsConst = true;
  } break;
  case SpvOpConstantTrue: {
    *condVal = true;
    *condIsConst = true;
  } break;
  case SpvOpLogicalNot: {
    (void)GetConstCondition(cInst->GetSingleWordInOperand(0),
        condVal, condIsConst);
    if (*condIsConst)
      *condVal = !*condVal;
  } break;
  default: {
    *condIsConst = false;
  } break;
  }
}

void AggressiveDCEPass::AddBranch(uint32_t labelId, ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranch(
    new ir::Instruction(SpvOpBranch, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  def_use_mgr_->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

void AggressiveDCEPass::KillAllInsts(ir::BasicBlock* bp) {
  bp->ForEachInst([this](ir::Instruction* ip) {
    def_use_mgr_->KillInst(ip);
  });
}

bool AggressiveDCEPass::GetConstConditionalBranch(ir::BasicBlock* bp,
    ir::Instruction** branchInst, ir::Instruction** mergeInst,
    bool *condVal) {
  auto ii = bp->end();
  --ii;
  *branchInst = &*ii;
  if ((*branchInst)->opcode() != SpvOpBranchConditional)
    return false;
  --ii;
  *mergeInst = &*ii;
  if ((*mergeInst)->opcode() != SpvOpSelectionMerge)
    return false;
  bool condIsConst;
  (void) GetConstCondition(
      (*branchInst)->GetSingleWordInOperand(kSpvBranchCondConditionalId),
      condVal, &condIsConst);
  return condIsConst;
}

bool AggressiveDCEPass::IsLocalVar(uint32_t varId) {
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = def_use_mgr_->GetDef(varTypeId);
  return varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) ==
      SpvStorageClassFunction;
}

void AggressiveDCEPass::AddStores(uint32_t ptrId) {
  analysis::UseList* uses = def_use_mgr_->GetUses(ptrId);
  if (uses == nullptr)
    return;
  for (auto u : *uses) {
    SpvOp op = u.inst->opcode();
    if (op == SpvOpStore)
      worklist_.push(u.inst);
    else
      AddStores(u.inst->result_id());
  }
  return false;
}

bool AggressiveDCEPass::AggressiveDCE(ir::Function* func) {
  // Map instruction to block. Add non-local store and return value
  // instructions to worklist. If function call encountered, return
  // false (unmodified).
  for (auto& blk : *func) {
    inst2block_[blk.GetLabelInst()] = &blk;
    for (auto& inst : blk) {
      inst2block_[&inst] = &blk;
      switch (inst.opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        (void) GetPtr(&inst, &varId);
        if (!IsLocalVar(varId)) {
          worklist_.push(&inst);
        }
      } break;
      case SpvOpReturnValue: {
        worklist_.push(&inst);
      } break;
      case SpvOpFunctionCall: {
        return false;
      } break;
      default:
        break;
      }
    }
  }
  // Create Control Dependence Tree
  std::list<ir::BasicBlock*> structuredOrder;
  ComputeStructuredOrder(func, &structuredOrder);
  std::stack<ir::BasicBlock*> currentHeaders;
  std::stack<uint32_t> currentMergeIds;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    if ((*bi)->id() == currentMergeIds.top()) {
      (void) currentHeaders.pop();
      (void) currentMergeIds.pop();
    }
    if (currentHeaders.empty())
      immediate_control_parent_[*bi] = nullptr;
    else
      immediate_control_parent_[*bi] = currentHeaders.top();
    uint32_t mergeId = MergeBlockIdIfAny(*bi);
    if (mergeId != 0) {
      currentHeaders.push(*bi);
      currentMergeIds.push(mergeId);
    }
  }
  // Perform closure on live instruction set. Also compute live blocks
  // and live control constructs.
  while (!worklist_.empty()) {
    ir::Instruction* liveInst = worklist_.front();
    live_insts_.insert(liveInst);
    // If block not yet live, mark it and its containing structure construct
    // headers live.
    ir::BasicBlock *liveBlock = inst2block_[liveInst];
    while (liveBlock != nullptr &&
        live_blocks_.find(liveBlock) == live_blocks_.end()) {
      live_blocks_.insert(liveBlock);
      liveBlock = immediate_control_parent[liveBlock];
    }
    // Add all operands if not already live
    liveInst->ForEachInId([this](const uint32_t* iid) {
      ir::Instruction* inInst = def_use_mgr_->GetDef(*iid);
      if (live_insts_.find(inInst) == live_insts_.end())
        worklist_.push(&inInst);
    });
    // If local load, add all variable's stores if variable not already live
    if (liveInst.opcode() == SpvOpLoad) {
      uint32_t varId;
      (void) GetPtr(&liveInst, &varId);
      if (IsLocalVar(varId)) {
        if (live_local_vars_.find(varId) == live_local_vars.end()) {
          AddStores(varId);
          live_local_vars_.insert(varId);
        }
      }
    }
    worklist_.pop();
  }
//
  std::list<ir::BasicBlock*> structuredOrder;
  ComputeStructuredOrder(func, &structuredOrder);
  std::unordered_set<ir::BasicBlock*> elimBlocks;
  bool modified = false;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    // Skip blocks that are already in the elimination set
    if (elimBlocks.find(*bi) != elimBlocks.end())
      continue;
    // Skip blocks that don't have constant conditional branch
    ir::Instruction* br;
    ir::Instruction* mergeInst;
    bool condVal;
    if (!GetConstConditionalBranch(*bi, &br, &mergeInst, &condVal))
      continue;

    // Replace conditional branch with unconditional branch
    uint32_t trueLabId = br->GetSingleWordInOperand(kSpvBranchCondTrueLabId);
    uint32_t falseLabId = br->GetSingleWordInOperand(kSpvBranchCondFalseLabId);
    uint32_t mergeLabId =
        mergeInst->GetSingleWordInOperand(kSpvSelectionMergeMergeBlockId);
    uint32_t liveLabId = condVal == true ? trueLabId : falseLabId;
    uint32_t deadLabId = condVal == true ? falseLabId : trueLabId;
    AddBranch(liveLabId, *bi);
    def_use_mgr_->KillInst(br);
    def_use_mgr_->KillInst(mergeInst);

    // Iterate to merge block deleting dead blocks
    std::unordered_set<uint32_t> deadLabIds;
    deadLabIds.insert(deadLabId);
    auto dbi = bi;
    ++dbi;
    uint32_t dLabId = (*dbi)->id();
    while (dLabId != mergeLabId) {
      if (deadLabIds.find(dLabId) != deadLabIds.end()) {
        // Add successor blocks to dead block set
        (*dbi)->ForEachSuccessorLabel([&deadLabIds](uint32_t succ) {
          deadLabIds.insert(succ);
        });
        // Add merge block to dead block set in case it has
        // no predecessors.
        uint32_t dMergeLabId = MergeBlockIdIfAny(**dbi);
        if (dMergeLabId != 0)
          deadLabIds.insert(dMergeLabId);
        // Kill use/def for all instructions and delete block
        KillAllInsts(*dbi);
        elimBlocks.insert(*dbi);
      }
      ++dbi;
      dLabId = (*dbi)->id();
    }

    // Process phi instructions in merge block.
    // deadLabIds are now blocks which cannot precede merge block.
    // If eliminated branch is to merge label, add current block to dead blocks.
    if (deadLabId == mergeLabId)
      deadLabIds.insert((*bi)->id());
    (*dbi)->ForEachPhiInst([&deadLabIds, this](ir::Instruction* phiInst) {
      uint32_t phiLabId0 = phiInst->GetSingleWordInOperand(kSpvPhiLab0Id);
      bool useFirst = deadLabIds.find(phiLabId0) == deadLabIds.end();
      uint32_t phiValIdx = useFirst ? kSpvPhiVal0Id : kSpvPhiVal1Id;
      uint32_t replId = phiInst->GetSingleWordInOperand(phiValIdx);
      uint32_t phiId = phiInst->result_id();
      (void)def_use_mgr_->ReplaceAllUsesWith(phiId, replId);
      def_use_mgr_->KillInst(phiInst);
    });
    modified = true;
  }

  // Erase dead blocks
  for (auto ebi = func->begin(); ebi != func->end(); )
    if (elimBlocks.find(&*ebi) != elimBlocks.end())
      ebi = ebi.Erase();
    else
      ++ebi;
  return modified;
}

void AggressiveDCEPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  id2block_.clear();
  block2structured_succs_.clear();
  for (auto& fn : *module_) {
    // Initialize function and block maps.
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
  }

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));
};

Pass::Status AggressiveDCEPass::ProcessImpl() {
  // Current functionality assumes structured control flow. 
  // TODO(): Handle non-structured control-flow.
  if (!module_->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;

  // Current functionality assumes logical addressing only
  // TODO(): Handle non-logical addressing
  if (module_->HasCapability(SpvCapabilityAddressing))
    return Status::SuccessWithoutChange;

  bool modified = false;
  // Call Mem2Reg on all remaining functions.
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || AggressiveDCEranches(fn);
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

AggressiveDCEPass::AggressiveDCEPass()
    : module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status AggressiveDCEPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

