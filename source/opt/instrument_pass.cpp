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

#include "instrument_pass.h"

#include "cfa.h"

// Debug Buffer Offsets
static const int kDebugOutputSizeOffset = 0;
static const int kDebugOutputDataOffset = 1;

// Common Output Record Offsets
static const int kInstCommonOutSize = 0;
static const int kInstCommonOutShaderId = 1;
static const int kInstCommonOutFunctionIdx = 2;
static const int kInstCommonOutInstructionIdx = 3;
static const int kInstCommonOutStageIdx = 4;

// Frag Shader Output Record Offsets
static const int kInstFragOutFragCoordX = 5;
static const int kInstFragOutFragCoordY = 6;
static const int kInstFragOutFragCoordZ = 7;
static const int kInstFragOutFragCoordW = 8;
static const int kInstFragOutRecordSize = 9;

// Indices of operands in SPIR-V instructions
static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFunctionCallArgumentId = 3;
static const int kSpvReturnValueId = 0;
static const int kSpvLoopMergeMergeBlockId = 0;
static const int kSpvLoopMergeContinueTargetIdInIdx = 1;
static const int kEntryPointExecutionModelInIdx = 0;
static const int kEntryPointFunctionIdInIdx = 1;
static const int kEntryPointInterfaceInIdx = 3;
static const int kSpvDecorateTargetIdInIdx = 0;
static const int kSpvDecorateDecorationInIdx = 1;
static const int kSpvDecorateBuiltinInIdx = 2;
static const int kSpvMemberDecorateDecorationInIdx = 2;
static const int kSpvMemberDecorateBuiltinInIdx = 3;

namespace spvtools {
namespace opt {

void InstrumentPass::MovePreludeCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  preCallSB_.clear();
  postCallSB_.clear();
  // Reuse label from ref block. Kill previous label
  // before reusing.
  uint32_t ref_blk_id = ref_block_itr->id();
  context()->KillInst(ref_block_itr->GetLabelInst());
  new_blk_ptr->reset(new BasicBlock(NewLabel(ref_blk_id)));
  // Move contents of original ref block up to ref instruction.
  for (auto cii = ref_block_itr->begin(); cii != ref_inst_itr;
      cii = ref_block_itr->begin()) {
    Instruction* inst = &*cii;
    inst->RemoveFromList();
    std::unique_ptr<Instruction> mv_ptr(inst);
    // Remember same-block ops for possible regeneration.
    if (IsSameBlockOp(&*mv_ptr)) {
      auto* sb_inst_ptr = mv_ptr.get();
      preCallSB_[mv_ptr->result_id()] = sb_inst_ptr;
    }
    (*new_blk_ptr)->AddInstruction(std::move(mv_ptr));
  }
}

void InstrumentPass::MovePostludeCode(
  UptrVectorIterator<BasicBlock> ref_block_itr,
  std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // new_blk_ptr->reset(new BasicBlock(NewLabel(ref_block_itr->id())));
  // Move contents of original ref block.
  for (auto cii = ref_block_itr->begin(); cii != ref_block_itr->end();
      cii = ref_block_itr->begin()) {
    Instruction* inst = &*cii;
    inst->RemoveFromList();
    std::unique_ptr<Instruction> mv_inst(inst);
    // Regenerate any same-block instruction that has not been seen in the
    // current block.
    if (preCallSB_.size() > 0) {
      CloneSameBlockOps(&mv_inst, &postCallSB_, &preCallSB_, new_blk_ptr);
      // Remember same-block ops in this block.
      if (IsSameBlockOp(&*mv_inst)) {
        const uint32_t rid = mv_inst->result_id();
        postCallSB_[rid] = rid;
      }
    }
    (*new_blk_ptr)->AddInstruction(std::move(mv_inst));
  }
}

void InstrumentPass::AddUnaryOp(uint32_t type_id, uint32_t result_id,
                                     SpvOp opcode, uint32_t operand,
                                     std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newUnOp(
      new Instruction(context(), opcode, type_id, result_id,
      { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newUnOp);
  (*block_ptr)->AddInstruction(std::move(newUnOp));
}

void InstrumentPass::AddBinaryOp(uint32_t type_id, uint32_t result_id,
  SpvOp opcode, uint32_t operand1, uint32_t operand2,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBinOp(
    new Instruction(context(), opcode, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand2 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBinOp);
  (*block_ptr)->AddInstruction(std::move(newBinOp));
}

void InstrumentPass::AddTernaryOp(uint32_t type_id, uint32_t result_id,
  SpvOp opcode, uint32_t operand1, uint32_t operand2, uint32_t operand3,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newTernOp(
    new Instruction(context(), opcode, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand2 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand3 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newTernOp);
  (*block_ptr)->AddInstruction(std::move(newTernOp));
}

void InstrumentPass::AddQuadOp(uint32_t type_id, uint32_t result_id,
  SpvOp opcode, uint32_t operand1, uint32_t operand2, uint32_t operand3,
  uint32_t operand4, std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newQuadOp(
    new Instruction(context(), opcode, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand2 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand3 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand4 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newQuadOp);
  (*block_ptr)->AddInstruction(std::move(newQuadOp));
}

void InstrumentPass::AddExtractOp(uint32_t type_id, uint32_t result_id,
  uint32_t operand1, uint32_t operand2,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBinOp(
    new Instruction(context(), SpvOpCompositeExtract, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,{ operand2 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBinOp);
  (*block_ptr)->AddInstruction(std::move(newBinOp));
}

void InstrumentPass::AddArrayLength(uint32_t result_id,
    uint32_t struct_ptr_id, uint32_t member_idx,
    std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newALenOp(
      new Instruction(context(), SpvOpArrayLength,
          GetTypeId(&analysis::Integer(32, false)), result_id,
          { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ struct_ptr_id } },
            { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {
              member_idx } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newALenOp);
  (*block_ptr)->AddInstruction(std::move(newALenOp));
}

uint32_t InstrumentPass::FindBuiltin(uint32_t builtin_val) {
  for (auto& a : get_module()->annotations()) {
    if (a.opcode() == SpvOpDecorate) {
      if (a.GetSingleWordInOperand(kSpvDecorateDecorationInIdx) !=
          SpvDecorationBuiltIn)
        continue;
      if (a.GetSingleWordInOperand(kSpvDecorateBuiltinInIdx) != builtin_val)
        continue;
      uint32_t target_id = a.GetSingleWordInOperand(kSpvDecorateTargetIdInIdx);
      Instruction* b_var = context()->get_def_use_mgr()->GetDef(target_id);
      // TODO(greg-lunarg): Support group builtins
      assert(b_var->opcode() == SpvOpVariable && "unexpected group builtin");
      return target_id;
    }
    else if (a.opcode() == SpvOpMemberDecorate) {
      if (a.GetSingleWordInOperand(kSpvMemberDecorateDecorationInIdx) !=
          SpvDecorationBuiltIn)
        continue;
      if (a.GetSingleWordInOperand(kSpvMemberDecorateBuiltinInIdx) !=
          builtin_val)
        continue;
      // TODO(greg-lunarg): Support member builtins
      assert(false && "unexpected member builtin");
      return 0;
    }
  }
  return 0;
}

void InstrumentPass::AddDecoration(uint32_t inst_id, uint32_t decoration,
    uint32_t decoration_value) {
  std::unique_ptr<Instruction> newDecoOp(
    new Instruction(context(), SpvOpDecorate, 0, 0,
      { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          { inst_id } },
        { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          { decoration } },
        { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          { decoration_value } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newDecoOp);
  get_decoration_mgr()->AddDecoration(&*newDecoOp);
  get_module()->AddAnnotationInst(std::move(newDecoOp));
}

void InstrumentPass::AddSelectionMerge(
  uint32_t mergeBlockId, uint32_t selControl,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newSMOp(
    new Instruction(context(), SpvOpSelectionMerge, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {mergeBlockId}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {selControl}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newSMOp);
  (*block_ptr)->AddInstruction(std::move(newSMOp));
}

void InstrumentPass::AddBranch(uint32_t label_id,
                           std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranch, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBranch);
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InstrumentPass::AddBranchCond(uint32_t cond_id, uint32_t true_id,
                               uint32_t false_id,
                               std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranchConditional, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {cond_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {true_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {false_id}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBranch);
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InstrumentPass::AddPhi(uint32_t type_id, uint32_t result_id, uint32_t var0_id,
  uint32_t parent0_id, uint32_t var1_id, uint32_t parent1_id,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newPhi(
    new Instruction(context(), SpvOpPhi, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ var0_id } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ parent0_id } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ var1_id } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ parent1_id } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newPhi);
  (*block_ptr)->AddInstruction(std::move(newPhi));
}

std::unique_ptr<Instruction> InstrumentPass::NewLabel(uint32_t label_id) {
  std::unique_ptr<Instruction> newLabel(
      new Instruction(context(), SpvOpLabel, 0, label_id, {}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newLabel);
  return newLabel;
}

uint32_t InstrumentPass::GetNullId(uint32_t type_id) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  const analysis::Type* type = type_mgr->GetType(type_id);
  const analysis::Constant* null_const = const_mgr->GetConstant(type, {});
  Instruction* null_inst =
      const_mgr->GetDefiningInstruction(null_const, type_id);
  return null_inst->result_id();
}

uint32_t InstrumentPass::GetUintConstantId(uint32_t u) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  analysis::Type* uint_ty = type_mgr->GetRegisteredType(
      &analysis::Integer(32, false));
  const analysis::Constant* uint_const = const_mgr->GetConstant(uint_ty, {u});
  Instruction* uint_inst = const_mgr->GetDefiningInstruction(uint_const);
  return uint_inst->result_id();
}

void InstrumentPass::GenDebugOutputFieldCode(
    uint32_t base_offset_id,
    uint32_t field_offset,
    uint32_t field_value_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  uint32_t data_idx_id = TakeNextId();
  AddBinaryOp(GetTypeId(&analysis::Integer(32, false)), data_idx_id, SpvOpIAdd,
      base_offset_id, GetUintConstantId(field_offset), new_blk_ptr);
  uint32_t achain_id = TakeNextId();
  AddTernaryOp(GetOutputBufferUintPtrId(), achain_id, SpvOpAccessChain,
      GetOutputBufferId(), GetUintConstantId(kDebugOutputDataOffset),
      data_idx_id, new_blk_ptr);
  AddBinaryOp(0, 0, SpvOpStore, achain_id, field_value_id, new_blk_ptr);
}

void InstrumentPass::GenCommonDebugOutputCode(
    uint32_t record_sz,
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    uint32_t base_offset_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // Store record size
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutSize,
      GetUintConstantId(record_sz), new_blk_ptr);
  // Store Shader Id
  // TODO(greg-lunarg): Get shader id from command argument
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutShaderId,
      GetUintConstantId(23), new_blk_ptr);
  // Store Function Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutFunctionIdx,
      GetUintConstantId(func_idx), new_blk_ptr);
  // Store Instruction Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutInstructionIdx,
      GetUintConstantId(instruction_idx), new_blk_ptr);
  // Store Stage Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutStageIdx,
      GetUintConstantId(stage_idx), new_blk_ptr);
}

void InstrumentPass::GenFragCoordEltDebugOutputCode(
    uint32_t base_offset_id,
    uint32_t uint_frag_coord_id,
    uint32_t element,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  uint32_t element_val_id = TakeNextId();
  AddExtractOp(GetTypeId(&analysis::Integer(32, false)), element_val_id,
      uint_frag_coord_id, element, new_blk_ptr);
  GenDebugOutputFieldCode(base_offset_id, kInstFragOutFragCoordX + element,
      element_val_id, new_blk_ptr);
}

void InstrumentPass::GenFragDebugOutputCode(
    uint32_t base_offset_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // Load FragCoord and convert to Uint
  uint32_t frag_coord_id = TakeNextId();
  AddUnaryOp(GetVec4FloatId(), frag_coord_id, SpvOpLoad, GetFragCoordId(),
      new_blk_ptr);
  uint32_t uint_frag_coord_id = TakeNextId();
  AddUnaryOp(GetVec4UintId(), uint_frag_coord_id, SpvOpBitcast, frag_coord_id,
      new_blk_ptr);
  for (uint32_t u = 0; u < 4u; ++u)
    GenFragCoordEltDebugOutputCode(base_offset_id,  uint_frag_coord_id, u,
        new_blk_ptr);
}

uint32_t InstrumentPass::GetStageOutputRecordSize() {
  // TODO(greg-lunarg): Add support for all stages
  // TODO(greg-lunarg): Assert fragment shader
  return kInstFragOutRecordSize;
}

void InstrumentPass::GenDebugOutputCode(
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    const std::vector<uint32_t> &validation_data,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // Gen test if debug output buffer size will not be exceeded.
  uint32_t obuf_record_sz = GetStageOutputRecordSize() +
      static_cast<uint32_t>(validation_data.size());
  uint32_t obuf_curr_sz_ac_id = TakeNextId();
  AddBinaryOp(GetOutputBufferUintPtrId(), obuf_curr_sz_ac_id, SpvOpAccessChain,
      GetOutputBufferId(), GetUintConstantId(kDebugOutputSizeOffset),
      new_blk_ptr);
  // Fetch the current debug buffer written size atomically, adding the
  // size of the record to be written.
  uint32_t obuf_curr_sz_id = TakeNextId();
  AddQuadOp(GetTypeId(&analysis::Integer(32, false)), obuf_curr_sz_id,
      SpvOpAtomicIAdd,
      obuf_curr_sz_ac_id,
      GetUintConstantId(SpvScopeInvocation),
      GetUintConstantId(SpvMemoryAccessMaskNone),
      GetUintConstantId(obuf_record_sz),
      new_blk_ptr);
  // Compute new written size
  uint32_t obuf_new_sz_id = TakeNextId();
  AddBinaryOp(GetTypeId(&analysis::Integer(32, false)), obuf_new_sz_id,
      SpvOpIAdd,
      obuf_curr_sz_id, GetUintConstantId(obuf_record_sz),
      new_blk_ptr);
  // Fetch the data bound
  uint32_t obuf_bnd_id = TakeNextId();
  AddArrayLength(obuf_bnd_id,
      GetOutputBufferId(),
      kDebugOutputDataOffset,
      new_blk_ptr);
  // Test that new written size is less than or equal to debug output
  // data bound
  uint32_t obuf_safe_id = TakeNextId();
  AddBinaryOp(GetTypeId(&analysis::Bool()), obuf_safe_id,
      SpvOpULessThanEqual, obuf_new_sz_id, obuf_bnd_id,
      new_blk_ptr);
  uint32_t mergeBlkId = TakeNextId();
  uint32_t writeBlkId = TakeNextId();
  std::unique_ptr<Instruction> mergeLabel(NewLabel(mergeBlkId));
  std::unique_ptr<Instruction> validLabel(NewLabel(writeBlkId));
  AddSelectionMerge(mergeBlkId, SpvSelectionControlMaskNone, new_blk_ptr);
  AddBranchCond(obuf_safe_id, writeBlkId, mergeBlkId, new_blk_ptr);
  // Close safety test block and gen write block
  new_blocks->push_back(std::move(*new_blk_ptr));
  new_blk_ptr->reset(new BasicBlock(std::move(validLabel)));
  GenCommonDebugOutputCode(obuf_record_sz, func_idx,
      instruction_idx, stage_idx, obuf_curr_sz_id,
      new_blk_ptr);
  // TODO(greg-lunarg): Add support for all stages
  uint32_t curr_record_offset = 0;
  if (stage_idx == SpvExecutionModelFragment) {
    GenFragDebugOutputCode(obuf_curr_sz_id, new_blk_ptr);
    curr_record_offset = kInstFragOutRecordSize;
  }
  else {
    assert(false && "unsupported stage");
  }
  // Gen writes of validation specific data
  for (auto vid : validation_data) {
    GenDebugOutputFieldCode(obuf_curr_sz_id, curr_record_offset,
        vid, new_blk_ptr);
    ++curr_record_offset;
  }
  // Close write block and gen merge block
  AddBranch(mergeBlkId, new_blk_ptr);
  new_blocks->push_back(std::move(*new_blk_ptr));
  new_blk_ptr->reset(new BasicBlock(std::move(mergeLabel)));
}

uint32_t InstrumentPass::AddPointerToType(uint32_t type_id,
  SpvStorageClass storage_class) {
  uint32_t resultId = TakeNextId();
  std::unique_ptr<Instruction> type_inst(
    new Instruction(context(), SpvOpTypePointer, 0, resultId,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
    { uint32_t(storage_class) } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ type_id } } }));
  context()->AddType(std::move(type_inst));
  analysis::Type* pointeeTy;
  std::unique_ptr<analysis::Pointer> pointerTy;
  std::tie(pointeeTy, pointerTy) =
    context()->get_type_mgr()->GetTypeAndPointerType(type_id,
      SpvStorageClassFunction);
  context()->get_type_mgr()->RegisterType(resultId, *pointerTy);
  return resultId;
}

void InstrumentPass::AddLoopMerge(uint32_t merge_id, uint32_t continue_id,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newLoopMerge(new Instruction(
    context(), SpvOpLoopMerge, 0, 0,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ merge_id } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ continue_id } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_LOOP_CONTROL,{ 0 } } }));
  (*block_ptr)->AddInstruction(std::move(newLoopMerge));
}

void InstrumentPass::AddStore(uint32_t ptr_id, uint32_t val_id,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newStore(
    new Instruction(context(), SpvOpStore, 0, 0,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ ptr_id } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ val_id } } }));
  (*block_ptr)->AddInstruction(std::move(newStore));
}

void InstrumentPass::AddLoad(uint32_t type_id, uint32_t resultId, uint32_t ptr_id,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newLoad(
    new Instruction(context(), SpvOpLoad, type_id, resultId,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ ptr_id } } }));
  (*block_ptr)->AddInstruction(std::move(newLoad));
}

uint32_t InstrumentPass::GetFalseId() {
  if (false_id_ != 0) return false_id_;
  false_id_ = get_module()->GetGlobalValue(SpvOpConstantFalse);
  if (false_id_ != 0) return false_id_;
  uint32_t boolId = get_module()->GetGlobalValue(SpvOpTypeBool);
  if (boolId == 0) {
    boolId = TakeNextId();
    get_module()->AddGlobalValue(SpvOpTypeBool, boolId, 0);
  }
  false_id_ = TakeNextId();
  get_module()->AddGlobalValue(SpvOpConstantFalse, false_id_, boolId);
  return false_id_;
}

void InstrumentPass::MapParams(
    Function* calleeFn, BasicBlock::iterator call_inst_itr,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  int param_idx = 0;
  calleeFn->ForEachParam([&call_inst_itr, &param_idx,
                          &callee2caller](const Instruction* cpi) {
    const uint32_t pid = cpi->result_id();
    (*callee2caller)[pid] = call_inst_itr->GetSingleWordOperand(
        kSpvFunctionCallArgumentId + param_idx);
    ++param_idx;
  });
}

void InstrumentPass::CloneAndMapLocals(
    Function* calleeFn, std::vector<std::unique_ptr<Instruction>>* new_vars,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  auto callee_block_itr = calleeFn->begin();
  auto callee_var_itr = callee_block_itr->begin();
  while (callee_var_itr->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<Instruction> var_inst(callee_var_itr->Clone(context()));
    uint32_t newId = TakeNextId();
    get_decoration_mgr()->CloneDecorations(callee_var_itr->result_id(), newId);
    var_inst->SetResultId(newId);
    (*callee2caller)[callee_var_itr->result_id()] = newId;
    new_vars->push_back(std::move(var_inst));
    ++callee_var_itr;
  }
}

uint32_t InstrumentPass::CreateReturnVar(
    Function* calleeFn, std::vector<std::unique_ptr<Instruction>>* new_vars) {
  uint32_t returnVarId = 0;
  const uint32_t calleeTypeId = calleeFn->type_id();
  analysis::Type* calleeType = context()->get_type_mgr()->GetType(calleeTypeId);
  if (calleeType->AsVoid() == nullptr) {
    // Find or create ptr to callee return type.
    uint32_t returnVarTypeId = context()->get_type_mgr()->FindPointerToType(
        calleeTypeId, SpvStorageClassFunction);
    if (returnVarTypeId == 0)
      returnVarTypeId = AddPointerToType(calleeTypeId, SpvStorageClassFunction);
    // Add return var to new function scope variables.
    returnVarId = TakeNextId();
    std::unique_ptr<Instruction> var_inst(
        new Instruction(context(), SpvOpVariable, returnVarTypeId, returnVarId,
                        {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
                          {SpvStorageClassFunction}}}));
    new_vars->push_back(std::move(var_inst));
  }
  get_decoration_mgr()->CloneDecorations(calleeFn->result_id(), returnVarId);
  return returnVarId;
}

bool InstrumentPass::IsSameBlockOp(const Instruction* inst) const {
  return inst->opcode() == SpvOpSampledImage || inst->opcode() == SpvOpImage;
}

void InstrumentPass::CloneSameBlockOps(
    std::unique_ptr<Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* postCallSB,
    std::unordered_map<uint32_t, Instruction*>* preCallSB,
    std::unique_ptr<BasicBlock>* block_ptr) {
  (*inst)->ForEachInId(
      [&postCallSB, &preCallSB, &block_ptr, this](uint32_t* iid) {
        const auto mapItr = (*postCallSB).find(*iid);
        if (mapItr == (*postCallSB).end()) {
          const auto mapItr2 = (*preCallSB).find(*iid);
          if (mapItr2 != (*preCallSB).end()) {
            // Clone pre-call same-block ops, map result id.
            const Instruction* inInst = mapItr2->second;
            std::unique_ptr<Instruction> sb_inst(inInst->Clone(context()));
            CloneSameBlockOps(&sb_inst, postCallSB, preCallSB, block_ptr);
            const uint32_t rid = sb_inst->result_id();
            const uint32_t nid = this->TakeNextId();
            get_decoration_mgr()->CloneDecorations(rid, nid);
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

void InstrumentPass::GenInstrumentCode(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<Instruction>>* new_vars,
    BasicBlock::iterator call_inst_itr,
    UptrVectorIterator<BasicBlock> call_block_itr) {
  // Map from all ids in the callee to their equivalent id in the caller
  // as callee instructions are copied into caller.
  std::unordered_map<uint32_t, uint32_t> callee2caller;
  // Pre-call same-block insts
  std::unordered_map<uint32_t, Instruction*> preCallSB;
  // Post-call same-block op ids
  std::unordered_map<uint32_t, uint32_t> postCallSB;

  // Invalidate the def-use chains.  They are not kept up to date while
  // inlining.  However, certain calls try to keep them up-to-date if they are
  // valid.  These operations can fail.
  context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);

  Function* calleeFn = id2function_[call_inst_itr->GetSingleWordOperand(
      kSpvFunctionCallFunctionId)];

  // Check for multiple returns in the callee.
  auto fi = multi_return_funcs_.find(calleeFn->result_id());
  const bool multiReturn = fi != multi_return_funcs_.end();

  // Map parameters to actual arguments.
  MapParams(calleeFn, call_inst_itr, &callee2caller);

  // Define caller local variables for all callee variables and create map to
  // them.
  CloneAndMapLocals(calleeFn, new_vars, &callee2caller);

  // Create return var if needed.
  uint32_t returnVarId = CreateReturnVar(calleeFn, new_vars);

  // Create set of callee result ids. Used to detect forward references
  std::unordered_set<uint32_t> callee_result_ids;
  calleeFn->ForEachInst([&callee_result_ids](const Instruction* cpi) {
    const uint32_t rid = cpi->result_id();
    if (rid != 0) callee_result_ids.insert(rid);
  });

  // If the caller is in a single-block loop, and the callee has multiple
  // blocks, then the normal inlining logic will place the OpLoopMerge in
  // the last of several blocks in the loop.  Instead, it should be placed
  // at the end of the first block.  First determine if the caller is in a
  // single block loop.  We'll wait to move the OpLoopMerge until the end
  // of the regular inlining logic, and only if necessary.
  bool caller_is_single_block_loop = false;
  bool caller_is_loop_header = false;
  if (auto* loop_merge = call_block_itr->GetLoopMergeInst()) {
    caller_is_loop_header = true;
    caller_is_single_block_loop =
        call_block_itr->id() ==
        loop_merge->GetSingleWordInOperand(kSpvLoopMergeContinueTargetIdInIdx);
  }

  bool callee_begins_with_structured_header =
      (*(calleeFn->begin())).GetMergeInst() != nullptr;

  // Clone and map callee code. Copy caller block code to beginning of
  // first block and end of last block.
  bool prevInstWasReturn = false;
  uint32_t singleTripLoopHeaderId = 0;
  uint32_t singleTripLoopContinueId = 0;
  uint32_t returnLabelId = 0;
  bool multiBlocks = false;
  const uint32_t calleeTypeId = calleeFn->type_id();
  // new_blk_ptr is a new basic block in the caller.  New instructions are
  // written to it.  It is created when we encounter the OpLabel
  // of the first callee block.  It is appended to new_blocks only when
  // it is complete.
  std::unique_ptr<BasicBlock> new_blk_ptr;
  calleeFn->ForEachInst([&new_blocks, &callee2caller, &call_block_itr,
                         &call_inst_itr, &new_blk_ptr, &prevInstWasReturn,
                         &returnLabelId, &returnVarId, caller_is_loop_header,
                         callee_begins_with_structured_header, &calleeTypeId,
                         &multiBlocks, &postCallSB, &preCallSB, multiReturn,
                         &singleTripLoopHeaderId, &singleTripLoopContinueId,
                         &callee_result_ids, this](const Instruction* cpi) {
    switch (cpi->opcode()) {
      case SpvOpFunction:
      case SpvOpFunctionParameter:
        // Already processed
        break;
      case SpvOpVariable:
        if (cpi->NumInOperands() == 2) {
          assert(callee2caller.count(cpi->result_id()) &&
                 "Expected the variable to have already been mapped.");
          uint32_t new_var_id = callee2caller.at(cpi->result_id());

          // The initializer must be a constant or global value.  No mapped
          // should be used.
          uint32_t val_id = cpi->GetSingleWordInOperand(1);
          AddStore(new_var_id, val_id, &new_blk_ptr);
        }
        break;
      case SpvOpUnreachable:
      case SpvOpKill: {
        // Generate a return label so that we split the block with the function
        // call. Copy the terminator into the new block.
        if (returnLabelId == 0) returnLabelId = this->TakeNextId();
        std::unique_ptr<Instruction> terminator(
            new Instruction(context(), cpi->opcode(), 0, 0, {}));
        new_blk_ptr->AddInstruction(std::move(terminator));
        break;
      }
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
        new_blk_ptr.reset(new BasicBlock(NewLabel(labelId)));
        if (firstBlock) {
          // Copy contents of original caller block up to call instruction.
          for (auto cii = call_block_itr->begin(); cii != call_inst_itr;
               cii = call_block_itr->begin()) {
            Instruction* inst = &*cii;
            inst->RemoveFromList();
            std::unique_ptr<Instruction> cp_inst(inst);
            // Remember same-block ops for possible regeneration.
            if (IsSameBlockOp(&*cp_inst)) {
              auto* sb_inst_ptr = cp_inst.get();
              preCallSB[cp_inst->result_id()] = sb_inst_ptr;
            }
            new_blk_ptr->AddInstruction(std::move(cp_inst));
          }
          if (caller_is_loop_header && callee_begins_with_structured_header) {
            // We can't place both the caller's merge instruction and another
            // merge instruction in the same block.  So split the calling block.
            // Insert an unconditional branch to a new guard block.  Later,
            // once we know the ID of the last block,  we will move the caller's
            // OpLoopMerge from the last generated block into the first block.
            // We also wait to avoid invalidating various iterators.
            const auto guard_block_id = this->TakeNextId();
            AddBranch(guard_block_id, &new_blk_ptr);
            new_blocks->push_back(std::move(new_blk_ptr));
            // Start the next block.
            new_blk_ptr.reset(new BasicBlock(NewLabel(guard_block_id)));
            // Reset the mapping of the callee's entry block to point to
            // the guard block.  Do this so we can fix up phis later on to
            // satisfy dominance.
            callee2caller[cpi->result_id()] = guard_block_id;
          }
          // If callee has multiple returns, insert a header block for
          // single-trip loop that will encompass callee code.  Start postheader
          // block.
          //
          // Note: Consider the following combination:
          //  - the caller is a single block loop
          //  - the callee does not begin with a structure header
          //  - the callee has multiple returns.
          // We still need to split the caller block and insert a guard block.
          // But we only need to do it once. We haven't done it yet, but the
          // single-trip loop header will serve the same purpose.
          if (multiReturn) {
            singleTripLoopHeaderId = this->TakeNextId();
            AddBranch(singleTripLoopHeaderId, &new_blk_ptr);
            new_blocks->push_back(std::move(new_blk_ptr));
            new_blk_ptr.reset(new BasicBlock(NewLabel(singleTripLoopHeaderId)));
            returnLabelId = this->TakeNextId();
            singleTripLoopContinueId = this->TakeNextId();
            AddLoopMerge(returnLabelId, singleTripLoopContinueId, &new_blk_ptr);
            uint32_t postHeaderId = this->TakeNextId();
            AddBranch(postHeaderId, &new_blk_ptr);
            new_blocks->push_back(std::move(new_blk_ptr));
            new_blk_ptr.reset(new BasicBlock(NewLabel(postHeaderId)));
            multiBlocks = true;
            // Reset the mapping of the callee's entry block to point to
            // the post-header block.  Do this so we can fix up phis later
            // on to satisfy dominance.
            callee2caller[cpi->result_id()] = postHeaderId;
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
        // If there was an early return, we generated a return label id
        // for it.  Now we have to generate the return block with that Id.
        if (returnLabelId != 0) {
          // If previous instruction was return, insert branch instruction
          // to return block.
          if (prevInstWasReturn) AddBranch(returnLabelId, &new_blk_ptr);
          if (multiReturn) {
            // If we generated a loop header to for the single-trip loop
            // to accommodate multiple returns, insert the continue
            // target block now, with a false branch back to the loop header.
            new_blocks->push_back(std::move(new_blk_ptr));
            new_blk_ptr.reset(
                new BasicBlock(NewLabel(singleTripLoopContinueId)));
            AddBranchCond(GetFalseId(), singleTripLoopHeaderId, returnLabelId,
                          &new_blk_ptr);
          }
          // Generate the return block.
          new_blocks->push_back(std::move(new_blk_ptr));
          new_blk_ptr.reset(new BasicBlock(NewLabel(returnLabelId)));
          multiBlocks = true;
        }
        // Load return value into result id of call, if it exists.
        if (returnVarId != 0) {
          const uint32_t resId = call_inst_itr->result_id();
          assert(resId != 0);
          AddLoad(calleeTypeId, resId, returnVarId, &new_blk_ptr);
        }
        // Copy remaining instructions from caller block.
        for (Instruction* inst = call_inst_itr->NextNode(); inst;
             inst = call_inst_itr->NextNode()) {
          inst->RemoveFromList();
          std::unique_ptr<Instruction> cp_inst(inst);
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
        // Finalize instrument code.
        new_blocks->push_back(std::move(new_blk_ptr));
      } break;
      default: {
        // Copy callee instruction and remap all input Ids.
        std::unique_ptr<Instruction> cp_inst(cpi->Clone(context()));
        cp_inst->ForEachInId([&callee2caller, &callee_result_ids,
                              this](uint32_t* iid) {
          const auto mapItr = callee2caller.find(*iid);
          if (mapItr != callee2caller.end()) {
            *iid = mapItr->second;
          } else if (callee_result_ids.find(*iid) != callee_result_ids.end()) {
            // Forward reference. Allocate a new id, map it,
            // use it and check for it when remapping result ids
            const uint32_t nid = this->TakeNextId();
            callee2caller[*iid] = nid;
            *iid = nid;
          }
        });
        // If result id is non-zero, remap it. If already mapped, use mapped
        // value, else use next id.
        const uint32_t rid = cp_inst->result_id();
        if (rid != 0) {
          const auto mapItr = callee2caller.find(rid);
          uint32_t nid;
          if (mapItr != callee2caller.end()) {
            nid = mapItr->second;
          } else {
            nid = this->TakeNextId();
            callee2caller[rid] = nid;
          }
          cp_inst->SetResultId(nid);
          get_decoration_mgr()->CloneDecorations(rid, nid);
        }
        new_blk_ptr->AddInstruction(std::move(cp_inst));
      } break;
    }
  });

  if (caller_is_loop_header && (new_blocks->size() > 1)) {
    // Move the OpLoopMerge from the last block back to the first, where
    // it belongs.
    auto& first = new_blocks->front();
    auto& last = new_blocks->back();
    assert(first != last);

    // Insert a modified copy of the loop merge into the first block.
    auto loop_merge_itr = last->tail();
    --loop_merge_itr;
    assert(loop_merge_itr->opcode() == SpvOpLoopMerge);
    std::unique_ptr<Instruction> cp_inst(loop_merge_itr->Clone(context()));
    if (caller_is_single_block_loop) {
      // Also, update its continue target to point to the last block.
      cp_inst->SetInOperand(kSpvLoopMergeContinueTargetIdInIdx, {last->id()});
    }
    first->tail().InsertBefore(std::move(cp_inst));

    // Remove the loop merge from the last block.
    loop_merge_itr->RemoveFromList();
    delete &*loop_merge_itr;
  }

  // Update block map given replacement blocks.
  for (auto& blk : *new_blocks) {
    id2block_[blk->id()] = &*blk;
  }
}

bool InstrumentPass::IsInlinableFunctionCall(const Instruction* inst) {
  if (inst->opcode() != SpvOp::SpvOpFunctionCall) return false;
  const uint32_t calleeFnId =
      inst->GetSingleWordOperand(kSpvFunctionCallFunctionId);
  const auto ci = inlinable_.find(calleeFnId);
  return ci != inlinable_.cend();
}

void InstrumentPass::UpdateSucceedingPhis(
    std::vector<std::unique_ptr<BasicBlock>>& new_blocks) {
  const auto firstBlk = new_blocks.begin();
  const auto lastBlk = new_blocks.end() - 1;
  const uint32_t firstId = (*firstBlk)->id();
  const uint32_t lastId = (*lastBlk)->id();
  const BasicBlock& const_last_block = *lastBlk->get();
  const_last_block.ForEachSuccessorLabel(
      [&firstId, &lastId, this](const uint32_t succ) {
        BasicBlock* sbp = this->id2block_[succ];
        sbp->ForEachPhiInst([&firstId, &lastId](Instruction* phi) {
          phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
            if (*id == firstId) *id = lastId;
          });
        });
      });
}

bool InstrumentPass::HasMultipleReturns(Function* func) {
  bool seenReturn = false;
  bool multipleReturns = false;
  for (auto& blk : *func) {
    auto terminal_ii = blk.cend();
    --terminal_ii;
    if (terminal_ii->opcode() == SpvOpReturn ||
        terminal_ii->opcode() == SpvOpReturnValue) {
      if (seenReturn) {
        multipleReturns = true;
        break;
      }
      seenReturn = true;
    }
  }
  return multipleReturns;
}

void InstrumentPass::ComputeStructuredSuccessors(Function* func) {
  // If header, make merge block first successor.
  for (auto& blk : *func) {
    uint32_t mbid = blk.MergeBlockIdIfAny();
    if (mbid != 0) {
      block2structured_succs_[&blk].push_back(id2block_[mbid]);
    }

    // Add true successors.
    const auto& const_blk = blk;
    const_blk.ForEachSuccessorLabel([&blk, this](const uint32_t sbid) {
      block2structured_succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}

InstrumentPass::GetBlocksFunction InstrumentPass::StructuredSuccessorsFunction() {
  return [this](const BasicBlock* block) {
    return &(block2structured_succs_[block]);
  };
}

bool InstrumentPass::HasNoReturnInLoop(Function* func) {
  // If control not structured, do not do loop/return analysis
  // TODO: Analyze returns in non-structured control flow
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return false;
  // Compute structured block order. This order has the property
  // that dominators are before all blocks they dominate and merge blocks
  // are after all blocks that are in the control constructs of their header.
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  std::list<const BasicBlock*> structuredOrder;
  CFA<BasicBlock>::DepthFirstTraversal(
      &*func->begin(), StructuredSuccessorsFunction(), ignore_block,
      [&](cbb_ptr b) { structuredOrder.push_front(b); }, ignore_edge);
  // Search for returns in loops. Only need to track outermost loop
  bool return_in_loop = false;
  uint32_t outerLoopMergeId = 0;
  for (auto& blk : structuredOrder) {
    // Exiting current outer loop
    if (blk->id() == outerLoopMergeId) outerLoopMergeId = 0;
    // Return block
    auto terminal_ii = blk->cend();
    --terminal_ii;
    if (terminal_ii->opcode() == SpvOpReturn ||
        terminal_ii->opcode() == SpvOpReturnValue) {
      if (outerLoopMergeId != 0) {
        return_in_loop = true;
        break;
      }
    } else if (terminal_ii != blk->cbegin()) {
      auto merge_ii = terminal_ii;
      --merge_ii;
      // Entering outermost loop
      if (merge_ii->opcode() == SpvOpLoopMerge && outerLoopMergeId == 0)
        outerLoopMergeId =
            merge_ii->GetSingleWordOperand(kSpvLoopMergeMergeBlockId);
    }
  }
  return !return_in_loop;
}

void InstrumentPass::AnalyzeReturns(Function* func) {
  // Look for multiple returns
  if (!HasMultipleReturns(func)) {
    no_return_in_loop_.insert(func->result_id());
    return;
  }
  multi_return_funcs_.insert(func->result_id());
  // If multiple returns, see if any are in a loop
  if (HasNoReturnInLoop(func)) no_return_in_loop_.insert(func->result_id());
}

bool InstrumentPass::IsInlinableFunction(Function* func) {
  // We can only instrument a function if it has blocks.
  if (func->cbegin() == func->cend()) return false;
  // Do not inline functions with returns in loops. Currently early return
  // functions are inlined by wrapping them in a one trip loop and implementing
  // the returns as a branch to the loop's merge block. However, this can only
  // done validly if the return was not in a loop in the original function.
  // Also remember functions with multiple (early) returns.
  AnalyzeReturns(func);
  return no_return_in_loop_.find(func->result_id()) !=
         no_return_in_loop_.cend();
}

// Return id for uint type
uint32_t InstrumentPass::GetTypeId(analysis::Type* ty_ptr) {
  return context()->get_type_mgr()->GetTypeInstruction(ty_ptr);
}

// Return id for output buffer uint ptr type
uint32_t InstrumentPass::GetOutputBufferUintPtrId() {
  if (output_buffer_uint_ptr_id_ == 0) {
    output_buffer_uint_ptr_id_ = context()->get_type_mgr()->FindPointerToType(
        GetTypeId(&analysis::Integer(32, false)), SpvStorageClassStorageBuffer);
  }
  return output_buffer_uint_ptr_id_;
}

uint32_t InstrumentPass::GetOutputBufferBinding() {
  switch (validation_id_) {
    case kInstValidationIdBindless: return kDebugOutputBindingBindless;
    default: assert(false && "unexpected validation id");
  }
  return 0;
}

// Return id for output buffer
uint32_t InstrumentPass::GetOutputBufferId() {
  if (output_buffer_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* uint_ty = type_mgr->GetRegisteredType(
        &analysis::Integer(32, false));
    analysis::Type* uint_rarr_ty = type_mgr->GetRegisteredType(
        &analysis::RuntimeArray(uint_ty));
    analysis::Type* obuf_ty = type_mgr->GetRegisteredType(
        &analysis::Struct({ uint_ty, uint_rarr_ty }));
    uint32_t obufTyId = type_mgr->GetTypeInstruction(obuf_ty);
    uint32_t obufTyPtrId_ = type_mgr->FindPointerToType(obufTyId,
        SpvStorageClassStorageBuffer);
    output_buffer_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(
        new Instruction(context(), SpvOpVariable, obufTyPtrId_,
            output_buffer_id_,
            { { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                { SpvStorageClassStorageBuffer } } }));
    get_def_use_mgr()->AnalyzeInstDefUse(&*newVarOp);
    get_module()->AddGlobalValue(std::move(newVarOp));
    // TODO(greg-lunarg): Get debug descriptor set from command argument
    AddDecoration(output_buffer_id_, SpvDecorationDescriptorSet, 7);
    AddDecoration(output_buffer_id_, SpvDecorationBinding,
        GetOutputBufferBinding());
  }
  return output_buffer_id_;
}

uint32_t InstrumentPass::GetVec4FloatId() {
  if (v4float_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* float_ty = type_mgr->GetRegisteredType(
        &analysis::Float(32));
    analysis::Type* v4float_ty = type_mgr->GetRegisteredType(
        &analysis::Vector(float_ty, 4));
    v4float_id_ = type_mgr->GetTypeInstruction(v4float_ty);
  }
  return v4float_id_;
}

uint32_t InstrumentPass::GetVec4UintId() {
  if (v4uint_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* uint_ty = type_mgr->GetRegisteredType(
        &analysis::Integer(32, false));
    analysis::Type* v4uint_ty = type_mgr->GetRegisteredType(
        &analysis::Vector(uint_ty, 4));
    v4uint_id_ = type_mgr->GetTypeInstruction(v4uint_ty);
  }
  return v4uint_id_;
}

uint32_t InstrumentPass::GetFragCoordId() {
  // If not yet known, look for one in shader
  if (frag_coord_id_ == 0) frag_coord_id_ = FindBuiltin(SpvBuiltInFragCoord);
  // If none in shader, create one
  if (frag_coord_id_ == 0) {
    uint32_t fragCoordTyPtrId = context()->get_type_mgr()->FindPointerToType(
      GetVec4FloatId(), SpvStorageClassInput);
    frag_coord_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(
      new Instruction(context(), SpvOpVariable, fragCoordTyPtrId,
        frag_coord_id_,
        { { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          { SpvStorageClassInput } } }));
    get_def_use_mgr()->AnalyzeInstDefUse(&*newVarOp);
    get_module()->AddGlobalValue(std::move(newVarOp));
    AddDecoration(frag_coord_id_, SpvDecorationBuiltIn, SpvBuiltInFragCoord);
  }
  return frag_coord_id_;
}

bool InstrumentPass::InstProcessCallTreeFromRoots(
  InstProcessFunction& pfn,
  const std::unordered_map<uint32_t, Function*>& id2function,
  std::queue<uint32_t>* roots,
  uint32_t stage_idx) {
  // Process call tree
  bool modified = false;
  std::unordered_set<uint32_t> done;

  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = id2function.at(fi);
      modified = pfn(fn, stage_idx) || modified;
      AddCalls(fn, roots);
    }
  }
  return modified;
}

bool InstrumentPass::InstProcessEntryPointCallTree(
    InstProcessFunction& pfn,
    Module* module) {
  // Map from function's result id to function
  std::unordered_map<uint32_t, Function*> id2function;
  for (auto& fn : *module) id2function[fn.result_id()] = &fn;

  // Process each of the entry points as a root.
  std::queue<uint32_t> roots;
  for (auto& e : module->entry_points()) {
    // TODO(greg-lunarg): Handle all stages. Currently only handling
    // fragment shaders. In particular, we will need
    // to clone any functions which are in the call trees of entrypoints
    // with differing execution models.
    if (e.GetSingleWordInOperand(kEntryPointExecutionModelInIdx) != 
        SpvExecutionModelFragment)
      continue;
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  bool modified = InstProcessCallTreeFromRoots(pfn, id2function, &roots,
      SpvExecutionModelFragment);
  // If any function is modified, add FragCoord to all entry points that
  // don't have it.
  if (modified) {
    uint32_t ocnt = 0;
    for (auto& e : module->entry_points()) {
      bool found = false;
      e.ForEachInOperand([&ocnt,&found,this](const uint32_t* idp){
        if (ocnt < kEntryPointInterfaceInIdx) return;
        if (*idp == this->GetFragCoordId()) found = true;
      });
      if (!found) e.AddOperand({ SPV_OPERAND_TYPE_ID, {GetFragCoordId()} });
    }
  }
  return modified;
}

void InstrumentPass::InitializeInstrument(uint32_t validation_id) {
  false_id_ = 0;
  validation_id_ = validation_id;
  output_buffer_id_ = 0;
  output_buffer_uint_ptr_id_ = 0;
  frag_coord_id_ = 0;
  v4float_id_ = 0;
  v4uint_id_ = 0;

  // clear collections
  id2function_.clear();
  id2block_.clear();
  block2structured_succs_.clear();
  inlinable_.clear();
  no_return_in_loop_.clear();
  multi_return_funcs_.clear();

  for (auto& fn : *get_module()) {
    // Initialize function and block maps.
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
    // Compute inlinability
    if (IsInlinableFunction(&fn)) inlinable_.insert(fn.result_id());
  }
}

InstrumentPass::InstrumentPass() {}

}  // namespace opt
}  // namespace spvtools
