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

#include "inline_pass.h"
#include "cfa.h"

// Indices of operands in SPIR-V instructions

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFunctionCallArgumentId = 3;
static const int kSpvReturnValueId = 0;
static const int kSpvTypePointerStorageClass = 1;
static const int kSpvTypePointerTypeId = 2;
static const int kSpvLoopMergeMergeBlockId = 0;
static const int kSpvSelectionMergeMergeBlockId = 0;
static const int kSpvBranchTargetLabelId = 0;

namespace spvtools {
namespace opt {

uint32_t InlinePass::FindPointerToType(uint32_t type_id,
                                       SpvStorageClass storage_class) {
  ir::Module::inst_iterator type_itr = module_->types_values_begin();
  for (; type_itr != module_->types_values_end(); ++type_itr) {
    const ir::Instruction* type_inst = &*type_itr;
    if (type_inst->opcode() == SpvOpTypePointer &&
        type_inst->GetSingleWordOperand(kSpvTypePointerTypeId) == type_id &&
        type_inst->GetSingleWordOperand(kSpvTypePointerStorageClass) ==
            storage_class)
      return type_inst->result_id();
  }
  return 0;
}

uint32_t InlinePass::AddPointerToType(uint32_t type_id,
                                      SpvStorageClass storage_class) {
  uint32_t resultId = TakeNextId();
  std::unique_ptr<ir::Instruction> type_inst(new ir::Instruction(
      SpvOpTypePointer, 0, resultId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
        {uint32_t(storage_class)}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {type_id}}}));
  module_->AddType(std::move(type_inst));
  return resultId;
}

void InlinePass::AddBranch(uint32_t label_id,
  std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newBranch(new ir::Instruction(
    SpvOpBranch, 0, 0,
    {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddBranchCond(uint32_t cond_id, uint32_t true_id,
  uint32_t false_id, std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newBranch(new ir::Instruction(
    SpvOpBranchConditional, 0, 0,
    {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {cond_id}},
     {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {true_id}},
     {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {false_id}}}));
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddLoopMerge(uint32_t merge_id, uint32_t continue_id,
                           std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newLoopMerge(new ir::Instruction(
      SpvOpLoopMerge, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {merge_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {continue_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LOOP_CONTROL, {0}}}));
  (*block_ptr)->AddInstruction(std::move(newLoopMerge));
}

void InlinePass::AddStore(uint32_t ptr_id, uint32_t val_id,
                          std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newStore(new ir::Instruction(
      SpvOpStore, 0, 0, {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}},
                         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {val_id}}}));
  (*block_ptr)->AddInstruction(std::move(newStore));
}

void InlinePass::AddLoad(uint32_t type_id, uint32_t resultId, uint32_t ptr_id,
                         std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newLoad(new ir::Instruction(
      SpvOpLoad, type_id, resultId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}}}));
  (*block_ptr)->AddInstruction(std::move(newLoad));
}

std::unique_ptr<ir::Instruction> InlinePass::NewLabel(uint32_t label_id) {
  std::unique_ptr<ir::Instruction> newLabel(
      new ir::Instruction(SpvOpLabel, 0, label_id, {}));
  return newLabel;
}

uint32_t InlinePass::GetFalseId() {
  if (falseId != 0)
    return falseId;
  falseId = module_->GetGlobalValue(SpvOpConstantFalse);
  if (falseId != 0)
    return falseId;
  uint32_t boolId = module_->GetGlobalValue(SpvOpTypeBool);
  if (boolId == 0) {
    boolId = TakeNextId();
    module_->AddGlobalValue(SpvOpTypeBool, boolId, 0);
  }
  falseId = TakeNextId();
  module_->AddGlobalValue(SpvOpConstantFalse, falseId, boolId);
  return falseId;
}

void InlinePass::MapParams(
    ir::Function* calleeFn,
    ir::UptrVectorIterator<ir::Instruction> call_inst_itr,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  int param_idx = 0;
  calleeFn->ForEachParam(
      [&call_inst_itr, &param_idx, &callee2caller](const ir::Instruction* cpi) {
        const uint32_t pid = cpi->result_id();
        (*callee2caller)[pid] = call_inst_itr->GetSingleWordOperand(
            kSpvFunctionCallArgumentId + param_idx);
        param_idx++;
      });
}

void InlinePass::CloneAndMapLocals(
    ir::Function* calleeFn,
    std::vector<std::unique_ptr<ir::Instruction>>* new_vars,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  auto callee_block_itr = calleeFn->begin();
  auto callee_var_itr = callee_block_itr->begin();
  while (callee_var_itr->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<ir::Instruction> var_inst(
        new ir::Instruction(*callee_var_itr));
    uint32_t newId = TakeNextId();
    var_inst->SetResultId(newId);
    (*callee2caller)[callee_var_itr->result_id()] = newId;
    new_vars->push_back(std::move(var_inst));
    callee_var_itr++;
  }
}

uint32_t InlinePass::CreateReturnVar(
    ir::Function* calleeFn,
    std::vector<std::unique_ptr<ir::Instruction>>* new_vars) {
  uint32_t returnVarId = 0;
  const uint32_t calleeTypeId = calleeFn->type_id();
  const ir::Instruction* calleeType =
      def_use_mgr_->id_to_defs().find(calleeTypeId)->second;
  if (calleeType->opcode() != SpvOpTypeVoid) {
    // Find or create ptr to callee return type.
    uint32_t returnVarTypeId =
        FindPointerToType(calleeTypeId, SpvStorageClassFunction);
    if (returnVarTypeId == 0)
      returnVarTypeId = AddPointerToType(calleeTypeId, SpvStorageClassFunction);
    // Add return var to new function scope variables.
    returnVarId = TakeNextId();
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(
        SpvOpVariable, returnVarTypeId, returnVarId,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
          {SpvStorageClassFunction}}}));
    new_vars->push_back(std::move(var_inst));
  }
  return returnVarId;
}

bool InlinePass::IsSameBlockOp(const ir::Instruction* inst) const {
  return inst->opcode() == SpvOpSampledImage || inst->opcode() == SpvOpImage;
}

void InlinePass::CloneSameBlockOps(
    std::unique_ptr<ir::Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* postCallSB,
    std::unordered_map<uint32_t, ir::Instruction*>* preCallSB,
    std::unique_ptr<ir::BasicBlock>* block_ptr) {
  (*inst)
      ->ForEachInId([&postCallSB, &preCallSB, &block_ptr, this](uint32_t* iid) {
        const auto mapItr = (*postCallSB).find(*iid);
        if (mapItr == (*postCallSB).end()) {
          const auto mapItr2 = (*preCallSB).find(*iid);
          if (mapItr2 != (*preCallSB).end()) {
            // Clone pre-call same-block ops, map result id.
            const ir::Instruction* inInst = mapItr2->second;
            std::unique_ptr<ir::Instruction> sb_inst(
                new ir::Instruction(*inInst));
            CloneSameBlockOps(&sb_inst, postCallSB, preCallSB, block_ptr);
            const uint32_t rid = sb_inst->result_id();
            const uint32_t nid = this->TakeNextId();
            sb_inst->SetResultId(nid);
            (*postCallSB)[rid] = nid;
            *iid = nid;
            (*block_ptr)->AddInstruction(std::move(sb_inst));
          }
        } else {
          // Reset same-block op operand.
          *iid = mapItr->second;
        }
      });
}

void InlinePass::GenInlineCode(
    std::vector<std::unique_ptr<ir::BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<ir::Instruction>>* new_vars,
    ir::UptrVectorIterator<ir::Instruction> call_inst_itr,
    ir::UptrVectorIterator<ir::BasicBlock> call_block_itr) {
  // Map from all ids in the callee to their equivalent id in the caller
  // as callee instructions are copied into caller.
  std::unordered_map<uint32_t, uint32_t> callee2caller;
  // Pre-call same-block insts
  std::unordered_map<uint32_t, ir::Instruction*> preCallSB;
  // Post-call same-block op ids
  std::unordered_map<uint32_t, uint32_t> postCallSB;

  ir::Function* calleeFn = id2function_[call_inst_itr->GetSingleWordOperand(
      kSpvFunctionCallFunctionId)];

  // Check for early returns
  auto fi = early_return_.find(calleeFn->result_id());
  bool earlyReturn = fi != early_return_.end();

  // Map parameters to actual arguments.
  MapParams(calleeFn, call_inst_itr, &callee2caller);

  // Define caller local variables for all callee variables and create map to
  // them.
  CloneAndMapLocals(calleeFn, new_vars, &callee2caller);

  // Create return var if needed.
  uint32_t returnVarId = CreateReturnVar(calleeFn, new_vars);

  // Clone and map callee code. Copy caller block code to beginning of
  // first block and end of last block.
  bool prevInstWasReturn = false;
  uint32_t headerId = 0;
  uint32_t continueId = 0;
  uint32_t returnLabelId = 0;
  bool multiBlocks = false;
  const uint32_t calleeTypeId = calleeFn->type_id();
  std::unique_ptr<ir::BasicBlock> new_blk_ptr;
  calleeFn->ForEachInst([&new_blocks, &callee2caller, &call_block_itr,
                         &call_inst_itr, &new_blk_ptr, &prevInstWasReturn,
                         &returnLabelId, &returnVarId, &calleeTypeId,
                         &multiBlocks, &postCallSB, &preCallSB, &earlyReturn,
                         &headerId, &continueId, this](
      const ir::Instruction* cpi) {
    switch (cpi->opcode()) {
      case SpvOpFunction:
      case SpvOpFunctionParameter:
      case SpvOpVariable:
        // Already processed
        break;
      case SpvOpLabel: {
        // If previous instruction was early return, insert branch
        // instruction to return block.
        if (prevInstWasReturn) {
          if (returnLabelId == 0) returnLabelId = this->TakeNextId();
          AddBranch(returnLabelId, &new_blk_ptr);
          prevInstWasReturn = false;
        }
        // Finish current block (if it exists) and get label for next block.
        uint32_t labelId;
        bool firstBlock = false;
        if (new_blk_ptr != nullptr) {
          new_blocks->push_back(std::move(new_blk_ptr));
          // If result id is already mapped, use it, otherwise get a new
          // one.
          const uint32_t rid = cpi->result_id();
          const auto mapItr = callee2caller.find(rid);
          labelId = (mapItr != callee2caller.end()) ? mapItr->second
                                                    : this->TakeNextId();
        } else {
          // First block needs to use label of original block
          // but map callee label in case of phi reference.
          labelId = call_block_itr->id();
          callee2caller[cpi->result_id()] = labelId;
          firstBlock = true;
        }
        // Create first/next block.
        new_blk_ptr.reset(new ir::BasicBlock(NewLabel(labelId)));
        if (firstBlock) {
          // Copy contents of original caller block up to call instruction.
          for (auto cii = call_block_itr->begin(); cii != call_inst_itr;
               cii++) {
            std::unique_ptr<ir::Instruction> cp_inst(new ir::Instruction(*cii));
            // Remember same-block ops for possible regeneration.
            if (IsSameBlockOp(&*cp_inst)) {
              auto* sb_inst_ptr = cp_inst.get();
              preCallSB[cp_inst->result_id()] = sb_inst_ptr;
            }
            new_blk_ptr->AddInstruction(std::move(cp_inst));
          }
          // If callee is early return function, insert header block for
          // one-trip loop that will encompass callee code. Start postheader
          // block.
          if (earlyReturn) {
            headerId = this->TakeNextId();
            AddBranch(headerId, &new_blk_ptr);
            new_blocks->push_back(std::move(new_blk_ptr));
            new_blk_ptr.reset(new ir::BasicBlock(NewLabel(headerId)));
            returnLabelId = this->TakeNextId();
            continueId = this->TakeNextId();
            AddLoopMerge(returnLabelId, continueId, &new_blk_ptr);
            uint32_t postHeaderId = this->TakeNextId();
            AddBranch(postHeaderId, &new_blk_ptr);
            new_blocks->push_back(std::move(new_blk_ptr));
            new_blk_ptr.reset(new ir::BasicBlock(NewLabel(postHeaderId)));
            multiBlocks = true;
          }
        } else {
          multiBlocks = true;
        }
      } break;
      case SpvOpReturnValue: {
        // Store return value to return variable.
        assert(returnVarId != 0);
        uint32_t valId = cpi->GetInOperand(kSpvReturnValueId).words[0];
        const auto mapItr = callee2caller.find(valId);
        if (mapItr != callee2caller.end()) {
          valId = mapItr->second;
        }
        AddStore(returnVarId, valId, &new_blk_ptr);

        // Remember we saw a return; if followed by a label, will need to
        // insert branch.
        prevInstWasReturn = true;
      } break;
      case SpvOpReturn: {
        // Remember we saw a return; if followed by a label, will need to
        // insert branch.
        prevInstWasReturn = true;
      } break;
      case SpvOpFunctionEnd: {
        // If there was an early return, insert continue and return blocks.
        // If previous instruction was return, insert branch instruction
        // to return block.
        if (returnLabelId != 0) {
          if (prevInstWasReturn) AddBranch(returnLabelId, &new_blk_ptr);
          new_blocks->push_back(std::move(new_blk_ptr));
          new_blk_ptr.reset(new ir::BasicBlock(NewLabel(continueId)));
          AddBranchCond(GetFalseId(), headerId, returnLabelId, &new_blk_ptr);
          new_blocks->push_back(std::move(new_blk_ptr));
          new_blk_ptr.reset(new ir::BasicBlock(NewLabel(returnLabelId)));
          multiBlocks = true;
        }
        // Load return value into result id of call, if it exists.
        if (returnVarId != 0) {
          const uint32_t resId = call_inst_itr->result_id();
          assert(resId != 0);
          AddLoad(calleeTypeId, resId, returnVarId, &new_blk_ptr);
        }
        // Copy remaining instructions from caller block.
        auto cii = call_inst_itr;
        for (cii++; cii != call_block_itr->end(); cii++) {
          std::unique_ptr<ir::Instruction> cp_inst(new ir::Instruction(*cii));
          // If multiple blocks generated, regenerate any same-block
          // instruction that has not been seen in this last block.
          if (multiBlocks) {
            CloneSameBlockOps(&cp_inst, &postCallSB, &preCallSB, &new_blk_ptr);
            // Remember same-block ops in this block.
            if (IsSameBlockOp(&*cp_inst)) {
              const uint32_t rid = cp_inst->result_id();
              postCallSB[rid] = rid;
            }
          }
          new_blk_ptr->AddInstruction(std::move(cp_inst));
        }
        // Finalize inline code.
        new_blocks->push_back(std::move(new_blk_ptr));
      } break;
      default: {
        // Copy callee instruction and remap all input Ids.
        std::unique_ptr<ir::Instruction> cp_inst(new ir::Instruction(*cpi));
        cp_inst->ForEachInId([&callee2caller, &cpi, this](uint32_t* iid) {
          const auto mapItr = callee2caller.find(*iid);
          if (mapItr != callee2caller.end()) {
            *iid = mapItr->second;
          } else if (cpi->has_labels()) {
            const ir::Instruction* inst =
                def_use_mgr_->id_to_defs().find(*iid)->second;
            if (inst->opcode() == SpvOpLabel) {
              // Forward label reference. Allocate a new label id, map it,
              // use it and check for it at each label.
              const uint32_t nid = this->TakeNextId();
              callee2caller[*iid] = nid;
              *iid = nid;
            }
          }
        });
        // Map and reset result id.
        const uint32_t rid = cp_inst->result_id();
        if (rid != 0) {
          const uint32_t nid = this->TakeNextId();
          callee2caller[rid] = nid;
          cp_inst->SetResultId(nid);
        }
        new_blk_ptr->AddInstruction(std::move(cp_inst));
      } break;
    }
  });
  // Update block map given replacement blocks.
  for (auto& blk : *new_blocks) {
    id2block_[blk->id()] = &*blk;
  }
}

bool InlinePass::IsInlinableFunctionCall(const ir::Instruction* inst) {
  if (inst->opcode() != SpvOp::SpvOpFunctionCall) return false;
  const uint32_t calleeFnId =
      inst->GetSingleWordOperand(kSpvFunctionCallFunctionId);
  const auto ci = inlinable_.find(calleeFnId);
  return ci != inlinable_.cend();
}

bool InlinePass::Inline(ir::Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (IsInlinableFunctionCall(&*ii)) {
        // Inline call.
        std::vector<std::unique_ptr<ir::BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<ir::Instruction>> newVars;
        GenInlineCode(&newBlocks, &newVars, ii, bi);
        // Update phi functions in successor blocks if call block
        // is replaced with more than one block.
        if (newBlocks.size() > 1) {
          const auto firstBlk = newBlocks.begin();
          const auto lastBlk = newBlocks.end() - 1;
          const uint32_t firstId = (*firstBlk)->id();
          const uint32_t lastId = (*lastBlk)->id();
          (*lastBlk)->ForEachSuccessorLabel(
              [&firstId, &lastId, this](uint32_t succ) {
                ir::BasicBlock* sbp = this->id2block_[succ];
                sbp->ForEachPhiInst([&firstId, &lastId](ir::Instruction* phi) {
                  phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
                    if (*id == firstId) *id = lastId;
                  });
                });
              });
        }
        // Replace old calling block with new block(s).
        bi = bi.Erase();
        bi = bi.InsertBefore(&newBlocks);
        // Insert new function variables.
        if (newVars.size() > 0) func->begin()->begin().InsertBefore(&newVars);
        // Restart inlining at beginning of calling block.
        ii = bi->begin();
        modified = true;
      } else {
        ii++;
      }
    }
  }
  return modified;
}

bool InlinePass::hasMultipleReturns(ir::Function* func) {
  bool seenReturn = false;
  bool multipleReturns = false;
  for (auto& blk : *func) {
    auto lii = blk.cend();
    lii--;
    if (lii->opcode() == SpvOpReturn || lii->opcode() == SpvOpReturnValue) {
      if (seenReturn) {
        multipleReturns = true;
        break;
      }
      seenReturn = true;
    }
  }
  return multipleReturns;
}

void InlinePass::ComputeStructuredSuccessors(ir::Function* func) {
  // If header, make merge block first successor.
  for (auto& blk : *func) {
    auto mii = blk.end();
    mii--;
    uint32_t mbid = 0;
    if (mii != blk.begin()) {
      mii--;
      if (mii->opcode() == SpvOpLoopMerge)
        uint32_t mbid = mii->GetSingleWordOperand(kSpvLoopMergeMergeBlockId);
      else if (mii->opcode() == SpvOpSelectionMerge)
        uint32_t mbid = mii->GetSingleWordOperand(kSpvSelectionMergeMergeBlockId);
    }
    if (mbid != 0)
      block2succs_[&blk].push_back(id2block_[mbid]);
    // add true successors
    blk.ForEachSuccessorLabel([&blk, this](uint32_t sbid) {
      block2succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}

InlinePass::GetBlocksFunction InlinePass::SuccessorsFunction() {
  return [this](const ir::BasicBlock* block) {
    return &(block2succs_[block]);
  };
}

bool InlinePass::hasNoReturnInLoop(ir::Function* func) {
  // If control not structured, do not do loop/return analysis
  // TODO: Analyze returns in non-structured control flow
  if (!module_->hasCapability(SpvCapabilityShader))
    return false;
  // Compute structured block order. This order has the property
  // that dominators are before all blocks they dominate and merge blocks
  // are after all blocks that are control dependent on their header.
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  std::list<const ir::BasicBlock*> structuredOrder;
  spvtools::CFA<ir::BasicBlock> cfa;
  cfa.DepthFirstTraversal(&*func->begin(), SuccessorsFunction(),
    ignore_block, [&](cbb_ptr b) { structuredOrder.push_front(b); }, ignore_edge);
  // Search for returns in loops. Only need to track outermost loop
  bool return_in_loop = false;
  uint32_t outerLoopMergeId = 0;
  for (auto& blk : structuredOrder) {
    // Exiting current outer loop
    if (blk->id() == outerLoopMergeId)
      outerLoopMergeId = 0;
    // Return block
    auto lii = blk->cend();
    lii--;
    if (lii->opcode() == SpvOpReturn || lii->opcode() == SpvOpReturnValue) {
      if (outerLoopMergeId != 0) {
        return_in_loop = true;
        break;
      }
    }
    else if (lii != blk->cbegin()) {
      auto mii = lii;
      mii--;
      // Entering outermost loop
      if (mii->opcode() == SpvOpLoopMerge && outerLoopMergeId == 0)
        outerLoopMergeId = mii->GetSingleWordOperand(kSpvLoopMergeMergeBlockId);
    }
  }
  return !return_in_loop;
}

void InlinePass::ReturnAnalysis(ir::Function* func) {
  // Look for multiple returns
  if (!hasMultipleReturns(func)) {
    no_return_in_loop_.insert(func->result_id());
    return;
  }
  early_return_.insert(func->result_id());
  // If multiple returns, see if any are in a loop
  if (hasNoReturnInLoop(func))
    no_return_in_loop_.insert(func->result_id());
}

bool InlinePass::IsInlinableFunction(ir::Function* func) {
  // We can only inline a function if it has blocks.
  if (func->cbegin() == func->cend())
    return false;
  // Do not inline functions with returns in loops. Currently early return
  // functions are inlined by wrapping them in a one trip loop and implementing
  // the returns as a branch to the loop's merge block. However, this can only
  // done validly if the return was not in a loop in the original function.
  // Also remember functions with multiple (early) returns.
  ReturnAnalysis(func);
  const auto ci = no_return_in_loop_.find(func->result_id());
  return ci != no_return_in_loop_.cend();
}

void InlinePass::Initialize(ir::Module* module) {
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

  // Initialize next unused Id.
  next_id_ = module->id_bound();

  // Save module.
  module_ = module;

  falseId = 0;

  id2function_.clear();
  id2block_.clear();
  block2succs_.clear();
  inlinable_.clear();
  for (auto& fn : *module_) {
    // Initialize function and block maps.
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
    // Compute inlinability
    if (IsInlinableFunction(&fn))
      inlinable_.insert(fn.result_id());
  }
};

Pass::Status InlinePass::ProcessImpl() {
  // Do exhaustive inlining on each entry point function in module
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || Inline(fn);
  }

  FinalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InlinePass::InlinePass()
    : module_(nullptr), def_use_mgr_(nullptr), next_id_(0) {}

Pass::Status InlinePass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
