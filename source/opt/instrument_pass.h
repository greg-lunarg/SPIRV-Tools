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

#ifndef LIBSPIRV_OPT_INSTRUMENT_PASS_H_
#define LIBSPIRV_OPT_INSTRUMENT_PASS_H_

#include <algorithm>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "decoration_manager.h"
#include "pass.h"

// Validation Ids
static const int kInstValidationIdBindless = 0;

// Debug Buffer Bindings
static const int kDebugOutputBindingBindless = 0;
static const int kDebugInputBindingBindless = 1;

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InstrumentPass : public Pass {
  using cbb_ptr = const BasicBlock*;

 public:
  using GetBlocksFunction =
      std::function<std::vector<BasicBlock*>*(const BasicBlock*)>;

  using InstProcessFunction = std::function<bool(Function*, uint32_t)>;

  virtual ~InstrumentPass() = default;

 protected:
  InstrumentPass();

  // Move all code in |ref_block_itr| preceding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePreludeCode(BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Move all code in |ref_block_itr| succeeding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePostludeCode(UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Return id for unsigned int constant value |u|.
  uint32_t GetUintConstantId(uint32_t u);
  
  void GenDebugOutputFieldCode(
    uint32_t base_offset_id,
    uint32_t field_offset,
    uint32_t field_value_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions which will write the fragment-shader-specific and
  // validation-specific members of the debug output buffer.
  void GenCommonDebugOutputCode(
    uint32_t record_sz,
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    uint32_t base_off,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  void GenFragCoordEltDebugOutputCode(
    uint32_t base_offset_id,
    uint32_t uint_frag_coord_id,
    uint32_t element,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions which will write the fragment-shader-specific and
  // validation-specific members of the debug output buffer.
  void GenFragDebugOutputCode(
    uint32_t base_off,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Return size of common and stage-specific output record members
  uint32_t GetStageOutputRecordSize();

  // Generate instructions which will write a record to the end of the debug
  // output buffer for the current shader.
  void GenDebugOutputCode(
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    const std::vector<uint32_t> &validation_data,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Add binary instruction |type_id, opcode, operand1, operand2| to
  // |block_ptr| and return resultId.
  void AddUnaryOp(
    uint32_t type_id, uint32_t result_id, SpvOp opcode,
    uint32_t operand, std::unique_ptr<BasicBlock>* block_ptr);

  // Add binary instruction |type_id, opcode, operand1, operand2| to
  // |block_ptr| and return resultId.
  void AddBinaryOp(
    uint32_t type_id, uint32_t result_id, SpvOp opcode,
    uint32_t operand1, uint32_t operand2,
    std::unique_ptr<BasicBlock>* block_ptr);

  // Add binary instruction |type_id, opcode, operand1, operand2| to
  // |block_ptr| and return resultId.
  void AddTernaryOp(
    uint32_t type_id, uint32_t result_id, SpvOp opcode,
    uint32_t operand1, uint32_t operand2, uint32_t operand3,
    std::unique_ptr<BasicBlock>* block_ptr);
  
  void AddQuadOp(uint32_t type_id, uint32_t result_id,
    SpvOp opcode, uint32_t operand1, uint32_t operand2, uint32_t operand3,
    uint32_t operand4, std::unique_ptr<BasicBlock>* block_ptr);

  // Add binary instruction |type_id, opcode, operand1, operand2| to
  // |block_ptr| and return resultId.
  void AddExtractOp(
    uint32_t type_id, uint32_t result_id,
    uint32_t operand1, uint32_t operand2,
    std::unique_ptr<BasicBlock>* block_ptr);

  void AddArrayLength(uint32_t result_id,
    uint32_t struct_ptr_id, uint32_t member_idx,
    std::unique_ptr<BasicBlock>* block_ptr);

  // Add SelectionMerge instruction |mergeBlockId, selectionControl| to
  // |block_ptr|.
  void AddSelectionMerge(
    uint32_t mergeBlockId, uint32_t selectionControl,
    std::unique_ptr<BasicBlock>* block_ptr);

  uint32_t FindBuiltin(uint32_t builtin_val);
  
  void AddDecoration(uint32_t inst_id, uint32_t decoration,
    uint32_t decoration_value);

  // Add unconditional branch to labelId to end of block block_ptr.
  void AddBranch(uint32_t labelId, std::unique_ptr<BasicBlock>* block_ptr);

  // Add conditional branch to end of block |block_ptr|.
  void AddBranchCond(uint32_t cond_id, uint32_t true_id, uint32_t false_id,
                     std::unique_ptr<BasicBlock>* block_ptr);

  void AddPhi(uint32_t type_id, uint32_t result_id, uint32_t var0_id,
              uint32_t parent0_id, uint32_t var1_id, uint32_t parent1_id,
              std::unique_ptr<BasicBlock>* block_ptr);

  // Return new label.
  std::unique_ptr<Instruction> NewLabel(uint32_t label_id);

  // Returns the id for the null constant value of |type_id|.
  uint32_t GetNullId(uint32_t type_id);

  // Return true if instruction must be in the same block that its result
  // is used.
  bool IsSameBlockOp(const Instruction* inst) const;

  // Clone operands which must be in same block as consumer instructions.
  // Look in preCallSB for instructions that need cloning. Look in
  // postCallSB for instructions already cloned. Add cloned instruction
  // to postCallSB.
  void CloneSameBlockOps(std::unique_ptr<Instruction>* inst,
                         std::unordered_map<uint32_t, uint32_t>* postCallSB,
                         std::unordered_map<uint32_t, Instruction*>* preCallSB,
                         std::unique_ptr<BasicBlock>* block_ptr);

  // Update phis in succeeding blocks to point to new last block
  void UpdateSucceedingPhis(
      std::vector<std::unique_ptr<BasicBlock>>& new_blocks);

  // Return id for |ty_ptr|
  uint32_t GetTypeId(analysis::Type* ty_ptr);

  // Return id for output buffer uint type
  uint32_t GetOutputBufferUintPtrId();
  
  uint32_t GetOutputBufferBinding();

  // Return id for output buffer
  uint32_t GetOutputBufferId();

  // Return id for FragCoord variable
  uint32_t GetFragCoordId();

  // Return id for v4float type
  uint32_t GetVec4FloatId();

  // Return id for v4uint type
  uint32_t GetVec4UintId();
  
  bool InstProcessCallTreeFromRoots(
    InstProcessFunction& pfn,
    const std::unordered_map<uint32_t, Function*>& id2function,
    std::queue<uint32_t>* roots,
    uint32_t stage_idx);

  bool InstProcessEntryPointCallTree(
    InstProcessFunction& pfn,
    Module* module);

  // Initialize state for optimization of |module|
  void InitializeInstrument(uint32_t validation_id);

  // Map from block's label id to block. TODO(dnovillo): This is superfluous wrt
  // CFG. It has functionality not present in CFG. Consolidate.
  std::unordered_map<uint32_t, BasicBlock*> id2block_;

  // result id for OpConstantFalse
  uint32_t validation_id_;

  // id for output buffer variable
  uint32_t output_buffer_id_;

  // type id for output buffer element
  uint32_t output_buffer_uint_ptr_id_;

  // id for FragCoord
  uint32_t frag_coord_id_;

  // id for v4float type
  uint32_t v4float_id_;

  // id for v4uint type
  uint32_t v4uint_id_;

  // Pre-instrumentation same-block insts
  std::unordered_map<uint32_t, Instruction*> preCallSB_;

  // Post-instrumentation same-block op ids
  std::unordered_map<uint32_t, uint32_t> postCallSB_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSTRUMENT_PASS_H_
