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

#include "inst_bindless_check_pass.h"

// Input Operand Indices
static const int kSpvImageSampleImageIdInIdx = 0;
static const int kSpvSampledImageImageIdInIdx = 0;
static const int kSpvSampledImageSamplerIdInIdx = 1;
static const int kSpvImageSampledImageIdInIdx = 0;
static const int kSpvLoadPtrIdInIdx = 0;
static const int kSpvAccessChainBaseIdInIdx = 0;
static const int kSpvAccessChainIndex0IdInIdx = 1;
static const int kSpvTypePointerTypeIdInIdx = 1;
static const int kSpvTypeArrayLengthIdInIdx = 1;
static const int kSpvConstantValueInIdx = 0;

// Bindless-specific Output Record Offsets
static const int kInstBindlessOutError = 0;
static const int kInstBindlessOutDescIndex = 1;
static const int kInstBindlessOutDescBound = 2;
static const int kInstBindlessOutRecordSize = 3;

namespace spvtools {
namespace opt {

void InstBindlessCheckPass::GenBindlessCheckCode(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<Instruction>>* new_vars,
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    uint32_t function_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx) {
  // Look for reference through bindless descriptor. If not, return.
  std::unique_ptr<BasicBlock> new_blk_ptr;
  uint32_t imageId;
  switch (ref_inst_itr->opcode()) {
    case SpvOp::SpvOpImageSampleImplicitLod:
    case SpvOp::SpvOpImageSampleExplicitLod:
    case SpvOp::SpvOpImageSampleDrefImplicitLod:
    case SpvOp::SpvOpImageSampleDrefExplicitLod:
    case SpvOp::SpvOpImageSampleProjImplicitLod:
    case SpvOp::SpvOpImageSampleProjExplicitLod:
    case SpvOp::SpvOpImageSampleProjDrefImplicitLod:
    case SpvOp::SpvOpImageSampleProjDrefExplicitLod:
    case SpvOp::SpvOpImageGather:
    case SpvOp::SpvOpImageDrefGather:
    case SpvOp::SpvOpImageQueryLod:
    case SpvOp::SpvOpImageSparseSampleImplicitLod:
    case SpvOp::SpvOpImageSparseSampleExplicitLod:
    case SpvOp::SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOp::SpvOpImageSparseSampleDrefExplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjImplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjExplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjDrefImplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjDrefExplicitLod:
    case SpvOp::SpvOpImageSparseGather:
    case SpvOp::SpvOpImageSparseDrefGather:
    case SpvOp::SpvOpImageFetch:
    case SpvOp::SpvOpImageRead:
    case SpvOp::SpvOpImageQueryFormat:
    case SpvOp::SpvOpImageQueryOrder:
    case SpvOp::SpvOpImageQuerySizeLod:
    case SpvOp::SpvOpImageQuerySize:
    case SpvOp::SpvOpImageQueryLevels:
    case SpvOp::SpvOpImageQuerySamples:
    case SpvOp::SpvOpImageSparseFetch:
    case SpvOp::SpvOpImageSparseRead:
    case SpvOp::SpvOpImageWrite:
      imageId =
          ref_inst_itr->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
      break;
    default:
      return;
  }
  Instruction* imageInst = get_def_use_mgr()->GetDef(imageId);
  uint32_t loadId;
  Instruction* loadInst;
  if (imageInst->opcode() == SpvOp::SpvOpSampledImage) {
    loadId = imageInst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    loadInst = get_def_use_mgr()->GetDef(loadId);
  }
  else if (imageInst->opcode() == SpvOp::SpvOpImage) {
    loadId = imageInst->GetSingleWordInOperand(kSpvImageSampledImageIdInIdx);
    loadInst = get_def_use_mgr()->GetDef(loadId);
  }
  else {
    loadId = imageId;
    loadInst = imageInst;
    imageId = 0;
  }
  if (loadInst->opcode() != SpvOp::SpvOpLoad) {
    assert(false && "expected image load");
    return;
  }
  uint32_t ptrId = loadInst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
  Instruction* ptrInst = get_def_use_mgr()->GetDef(ptrId);
  // Check descriptor index against upper bound
  // TODO(greg-lunarg): Check descriptor to make sure it is written.
  if (ptrInst->opcode() != SpvOp::SpvOpAccessChain)
    return;
  if (ptrInst->NumInOperands() != 2) {
    assert(false && "unexpected bindless index number");
    return;
  }
  uint32_t indexId =
      ptrInst->GetSingleWordInOperand(kSpvAccessChainIndex0IdInIdx);
  Instruction* indexInst = get_def_use_mgr()->GetDef(indexId);
  ptrId = ptrInst->GetSingleWordInOperand(kSpvAccessChainBaseIdInIdx);
  ptrInst = get_def_use_mgr()->GetDef(ptrId);
  if (ptrInst->opcode() != SpvOpVariable) {
    assert(false && "unexpected bindless base");
    return;
  }
  uint32_t varTypeId = ptrInst->type_id();
  Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  uint32_t ptrTypeId =
      varTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeIdInIdx);
  Instruction* ptrTypeInst = get_def_use_mgr()->GetDef(ptrTypeId);
  // TODO(greg-lunarg): Check descriptor index against runtime array
  // size.
  if (ptrTypeInst->opcode() != SpvOpTypeArray)
    return;
  // If index and bound both compile-time constants and index >= bound,
  // generate debug output error code and use zero as referenced value.
  uint32_t error_id = GetUintConstantId(kInstErrorBindlessBounds);
  uint32_t lengthId =
    ptrTypeInst->GetSingleWordInOperand(kSpvTypeArrayLengthIdInIdx);
  Instruction* lengthInst = get_def_use_mgr()->GetDef(lengthId);
  if (indexInst->opcode() == SpvOpConstant &&
      lengthInst->opcode() == SpvOpConstant) {
    if (indexInst->GetSingleWordInOperand(kSpvConstantValueInIdx) <
        lengthInst->GetSingleWordInOperand(kSpvConstantValueInIdx))
      return;
    MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
    GenDebugOutputCode(function_idx, instruction_idx,
        stage_idx, { error_id, indexId, lengthId }, new_blocks, new_vars,
        &new_blk_ptr);
    // Set the original reference id to zero, if it exists. Kill original
    // reference before reusing id.
    uint32_t ref_type_id = ref_inst_itr->type_id();
    uint32_t ref_result_id = ref_inst_itr->result_id();
    context()->KillInst(&*ref_inst_itr);
    if (ref_result_id != 0) {
      uint32_t nullId = GetNullId(ref_type_id);
      AddUnaryOp(ref_type_id, ref_result_id, SpvOpCopyObject,
          nullId, &new_blk_ptr);
    }
  }
  // Otherwise generate full runtime bounds test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  else {
    MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
    uint32_t ultId = TakeNextId();
    AddBinaryOp(GetBoolId(), ultId, SpvOpULessThan, indexId,
        lengthId, &new_blk_ptr);
    uint32_t mergeBlkId = TakeNextId();
    uint32_t validBlkId = TakeNextId();
    uint32_t invalidBlkId = TakeNextId();
    std::unique_ptr<Instruction> mergeLabel(NewLabel(mergeBlkId));
    std::unique_ptr<Instruction> validLabel(NewLabel(validBlkId));
    std::unique_ptr<Instruction> invalidLabel(NewLabel(invalidBlkId));
    AddSelectionMerge(mergeBlkId, SpvSelectionControlMaskNone, &new_blk_ptr);
    AddBranchCond(ultId, validBlkId, invalidBlkId, &new_blk_ptr);
    // Close selection block and gen valid reference block
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr.reset(new BasicBlock(std::move(validLabel)));
    // Clone descriptor load
    std::unique_ptr<Instruction> newLoadInst(loadInst->Clone(context()));
    uint32_t newLoadId = TakeNextId();
    newLoadInst->SetResultId(newLoadId);
    get_def_use_mgr()->AnalyzeInstDefUse(&*newLoadInst);
    new_blk_ptr->AddInstruction(std::move(newLoadInst));
    get_decoration_mgr()->CloneDecorations(loadInst->result_id(), newLoadId);
    uint32_t newImageId = newLoadId;
    // Clone Image/SampledImage with new load, if needed
    if (imageId != 0) {
      newImageId = TakeNextId();
      if (imageInst->opcode() == SpvOp::SpvOpSampledImage) {
        AddBinaryOp(imageInst->type_id(), newImageId, SpvOpSampledImage,
          newLoadId, imageInst->GetSingleWordInOperand(
            kSpvSampledImageSamplerIdInIdx), &new_blk_ptr);
      }
      else {
        assert(imageInst->opcode() == SpvOp::SpvOpImage && "expecting OpImage");
        AddUnaryOp(imageInst->type_id(), newImageId, SpvOpImage,
          newLoadId, &new_blk_ptr);
      }
      get_decoration_mgr()->CloneDecorations(imageId, newImageId);
    }
    // Clone original reference using new image code
    std::unique_ptr<Instruction> newRefInst(ref_inst_itr->Clone(context()));
    uint32_t ref_result_id = ref_inst_itr->result_id();
    uint32_t newRefId = 0;
    if (ref_result_id != 0) {
      newRefId = TakeNextId();
      newRefInst->SetResultId(newRefId);
    }
    newRefInst->SetInOperand(kSpvImageSampleImageIdInIdx, { newImageId });
    // Register new reference and add to new block
    get_def_use_mgr()->AnalyzeInstDefUse(&*newRefInst);
    new_blk_ptr->AddInstruction(std::move(newRefInst));
    if (newRefId != 0)
      get_decoration_mgr()->CloneDecorations(ref_result_id, newRefId);
    // Close valid block and gen invalid block
    AddBranch(mergeBlkId, &new_blk_ptr);
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr.reset(new BasicBlock(std::move(invalidLabel)));
    GenDebugOutputCode(function_idx, instruction_idx,
        stage_idx, { error_id, indexId, lengthId }, new_blocks, new_vars,
        &new_blk_ptr);
    // Remember last invalid block id
    uint32_t lastInvalidBlkId = new_blk_ptr->GetLabelInst()->result_id();
    // Gen zero for invalid  reference
    uint32_t ref_type_id = ref_inst_itr->type_id();
    uint32_t nullId = 0;
    if (newRefId != 0)
      nullId = GetNullId(ref_type_id);
    // Close invalid block and gen merge block
    AddBranch(mergeBlkId, &new_blk_ptr);
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr.reset(new BasicBlock(std::move(mergeLabel)));
    // Gen phi of new reference and zero, if necessary. Use result id of
    // original reference so we don't have to do a replace. Kill original
    // reference before reusing its id.
    context()->KillInst(&*ref_inst_itr);
    if (newRefId != 0)
      AddPhi(ref_type_id, ref_result_id, newRefId, validBlkId,
          nullId, lastInvalidBlkId, &new_blk_ptr);
  }
  MovePostludeCode(ref_block_itr, &new_blk_ptr);
  // Add remainder/merge block to new blocks
  new_blocks->push_back(std::move(new_blk_ptr));
}

bool InstBindlessCheckPass::InstBindlessCheck(Function* func, uint32_t stage_idx) {
  bool modified = false;
  // Compute function index
  uint32_t function_idx = 0;
  for (auto fii = get_module()->begin(); fii != get_module()->end(); ++fii) {
    if (&*fii == func)
      break;
    ++function_idx;
  }
  std::vector<std::unique_ptr<BasicBlock>> newBlocks;
  std::vector<std::unique_ptr<Instruction>> newVars;
  // Count function instruction
  uint32_t instruction_idx = 1;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    // Count block's label
    ++instruction_idx;
    for (auto ii = bi->begin(); ii != bi->end(); ++instruction_idx) {
      // Bump instruction count if debug instructions
      instruction_idx += static_cast<uint32_t>(ii->dbg_line_insts().size());
      // Generate bindless check if warranted
      GenBindlessCheckCode(&newBlocks, &newVars, ii, bi, function_idx,
          instruction_idx, stage_idx);
      if (newBlocks.size() == 0) {
        ++ii;
        continue;
      }
      // If there are new blocks we know there will always be two or
      // more, so update succeeding phis with label of new last block.
      size_t newBlocksSize = newBlocks.size();
      assert(newBlocksSize > 1);
      UpdateSucceedingPhis(newBlocks);
      // Replace original block with new block(s).
      bi = bi.Erase();
      for (auto& bb : newBlocks) {
        bb->SetParent(func);
      }
      bi = bi.InsertBefore(&newBlocks);
      // Reset block iterator to last new block
      for (size_t i = 0; i < newBlocksSize - 1; i++) ++bi;
      modified = true;
      // Restart instrumenting at beginning of last new block,
      // but skip over any new phi or copy instruction.
      ii = bi->begin();
      if (ii->opcode() == SpvOpPhi || ii->opcode() == SpvOpCopyObject) ++ii;
      newBlocks.clear();
    }
  }
  return modified;
}

void InstBindlessCheckPass::InitializeInstBindlessCheck() {
  // Initialize base class
  InitializeInstrument(kInstValidationIdBindless);
  // Look for related extensions
  ext_descriptor_indexing_defined_ = false;
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
      reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (strcmp(extName, "SPV_EXT_descriptor_indexing") == 0) {
      ext_descriptor_indexing_defined_ = true;
      break;
    }
  }
}

Pass::Status InstBindlessCheckPass::ProcessImpl() {
  // Perform instrumentation on each entry point function in module
  InstProcessFunction pfn = [this](Function* fp, uint32_t stage_idx) { 
      return InstBindlessCheck(fp, stage_idx); };
  bool modified = InstProcessEntryPointCallTree(pfn, get_module());
  // This pass does not update def/use info
  context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);
  // TODO(greg-lunarg): If modified, do CFGCleanup, DCE
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status InstBindlessCheckPass::Process() {
  InitializeInstBindlessCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
