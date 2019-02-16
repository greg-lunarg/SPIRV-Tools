// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

namespace {

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

}  // anonymous namespace

namespace spvtools {
namespace opt {

uint32_t InstBindlessCheckPass::GenDebugReadLength(
    uint32_t var_id, InstructionBuilder* builder) {
  uint32_t desc_set_idx =
      var2desc_set_[var_id] + kDebugInputBindlessOffsetLengths;
  uint32_t desc_set_idx_id = builder->GetUintConstantId(desc_set_idx);
  uint32_t binding_idx_id = builder->GetUintConstantId(var2binding_[var_id]);
  return GenDebugDirectRead({desc_set_idx_id, binding_idx_id}, builder);
}

uint32_t InstBindlessCheckPass::GenDebugReadInit(uint32_t var_id,
                                                 uint32_t desc_idx_id,
                                                 InstructionBuilder* builder) {
  uint32_t desc_set_base_id =
      builder->GetUintConstantId(kDebugInputBindlessInitOffset);
  uint32_t desc_set_idx_id = builder->GetUintConstantId(var2desc_set_[var_id]);
  uint32_t binding_idx_id = builder->GetUintConstantId(var2binding_[var_id]);
  uint32_t u_desc_idx_id = GenUintCastCode(desc_idx_id, builder);
  return GenDebugDirectRead(
      {desc_set_base_id, desc_set_idx_id, binding_idx_id, u_desc_idx_id},
      builder);
}

uint32_t InstBindlessCheckPass::CloneOriginalReference(
    InstructionBuilder* builder) {
  // Clone descriptor load
  Instruction* new_load_inst =
      builder->AddLoad(load_inst_->type_id(),
                       load_inst_->GetSingleWordInOperand(kSpvLoadPtrIdInIdx));
  uid2offset_[new_load_inst->unique_id()] =
      uid2offset_[load_inst_->unique_id()];
  uint32_t new_load_id = new_load_inst->result_id();
  get_decoration_mgr()->CloneDecorations(load_inst_->result_id(), new_load_id);
  uint32_t new_image_id = new_load_id;
  // Clone Image/SampledImage with new load, if needed
  if (image_id_ != 0) {
    if (image_inst_->opcode() == SpvOp::SpvOpSampledImage) {
      Instruction* new_image_inst = builder->AddBinaryOp(
          image_inst_->type_id(), SpvOpSampledImage, new_load_id,
          image_inst_->GetSingleWordInOperand(kSpvSampledImageSamplerIdInIdx));
      uid2offset_[new_image_inst->unique_id()] =
          uid2offset_[image_inst_->unique_id()];
      new_image_id = new_image_inst->result_id();
    } else {
      assert(image_inst_->opcode() == SpvOp::SpvOpImage && "expecting OpImage");
      Instruction* new_image_inst =
          builder->AddUnaryOp(image_inst_->type_id(), SpvOpImage, new_load_id);
      uid2offset_[new_image_inst->unique_id()] =
          uid2offset_[image_inst_->unique_id()];
      new_image_id = new_image_inst->result_id();
    }
    get_decoration_mgr()->CloneDecorations(image_id_, new_image_id);
  }
  // Clone original reference using new image code
  std::unique_ptr<Instruction> new_ref_inst(ref_inst_->Clone(context()));
  uint32_t ref_result_id = ref_inst_->result_id();
  uint32_t new_ref_id = 0;
  if (ref_result_id != 0) {
    new_ref_id = TakeNextId();
    new_ref_inst->SetResultId(new_ref_id);
  }
  new_ref_inst->SetInOperand(kSpvImageSampleImageIdInIdx, {new_image_id});
  // Register new reference and add to new block
  Instruction* added_inst = builder->AddInstruction(std::move(new_ref_inst));
  uid2offset_[added_inst->unique_id()] = uid2offset_[ref_inst_->unique_id()];
  if (new_ref_id != 0)
    get_decoration_mgr()->CloneDecorations(ref_result_id, new_ref_id);
  return new_ref_id;
}

uint32_t InstBindlessCheckPass::GetDescriptorValueId(Instruction* inst) {
  switch (inst->opcode()) {
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
      return inst->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
      break;
    default:
      break;
  }
  return 0;
}

bool InstBindlessCheckPass::AnalyzeDescriptorReference(Instruction* ref_inst) {
  image_id_ = GetDescriptorValueId(ref_inst);
  if (image_id_ == 0) return false;
  image_inst_ = get_def_use_mgr()->GetDef(image_id_);
  if (image_inst_->opcode() == SpvOp::SpvOpSampledImage) {
    load_id_ =
        image_inst_->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    load_inst_ = get_def_use_mgr()->GetDef(load_id_);
  } else if (image_inst_->opcode() == SpvOp::SpvOpImage) {
    load_id_ =
        image_inst_->GetSingleWordInOperand(kSpvImageSampledImageIdInIdx);
    load_inst_ = get_def_use_mgr()->GetDef(load_id_);
  } else {
    load_id_ = image_id_;
    load_inst_ = image_inst_;
    image_id_ = 0;
  }
  if (load_inst_->opcode() != SpvOp::SpvOpLoad) {
    // TODO(greg-lunarg): Handle additional possibilities?
    return false;
  }
  ptr_id_ = load_inst_->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
  ptr_inst_ = get_def_use_mgr()->GetDef(ptr_id_);
  if (ptr_inst_->opcode() == SpvOp::SpvOpVariable) {
    index_id_ = 0;
    var_id_ = ptr_id_;
    var_inst_ = ptr_inst_;
  } else if (ptr_inst_->opcode() == SpvOp::SpvOpAccessChain) {
    if (ptr_inst_->NumInOperands() != 2) {
      assert(false && "unexpected bindless index number");
      return false;
    }
    index_id_ = ptr_inst_->GetSingleWordInOperand(kSpvAccessChainIndex0IdInIdx);
    var_id_ = ptr_inst_->GetSingleWordInOperand(kSpvAccessChainBaseIdInIdx);
    var_inst_ = get_def_use_mgr()->GetDef(var_id_);
    if (var_inst_->opcode() != SpvOpVariable) {
      assert(false && "unexpected bindless base");
      return false;
    }
  } else {
    // TODO(greg-lunarg): Handle additional possibilities?
    return false;
  }
  ref_inst_ = ref_inst;
  return true;
}

void InstBindlessCheckPass::GenCheckCode(
    uint32_t check_id, uint32_t error_id, uint32_t length_id,
    uint32_t stage_idx, InstructionBuilder* builder,
    std::unique_ptr<BasicBlock>* new_blk_ptr,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Gen conditional branch on check_id. Valid branch generates original
  // reference. Invalid generates debug output and zero result (if needed).
  uint32_t merge_blk_id = TakeNextId();
  uint32_t valid_blk_id = TakeNextId();
  uint32_t invalid_blk_id = TakeNextId();
  std::unique_ptr<Instruction> merge_label(NewLabel(merge_blk_id));
  std::unique_ptr<Instruction> valid_label(NewLabel(valid_blk_id));
  std::unique_ptr<Instruction> invalid_label(NewLabel(invalid_blk_id));
  (void)builder->AddConditionalBranch(check_id, valid_blk_id, invalid_blk_id,
                                      merge_blk_id,
                                      SpvSelectionControlMaskNone);
  // Close selection block and gen valid bounds branch
  new_blocks->push_back(std::move(*new_blk_ptr));
  new_blk_ptr->reset(new BasicBlock(std::move(valid_label)));
  builder->SetInsertPoint(&**new_blk_ptr);
  uint32_t new_ref_id = CloneOriginalReference(builder);
  // Close valid bounds branch and gen invalid block
  (void)builder->AddBranch(merge_blk_id);
  new_blocks->push_back(std::move(*new_blk_ptr));
  new_blk_ptr->reset(new BasicBlock(std::move(invalid_label)));
  builder->SetInsertPoint(&**new_blk_ptr);
  uint32_t u_index_id = GenUintCastCode(index_id_, builder);
  GenDebugStreamWrite(uid2offset_[ref_inst_->unique_id()], stage_idx,
                      {error_id, u_index_id, length_id}, builder);
  // Remember last invalid block id
  uint32_t last_invalid_blk_id = (*new_blk_ptr)->GetLabelInst()->result_id();
  // Gen zero for invalid  reference
  uint32_t ref_type_id = ref_inst_->type_id();
  // Close invalid block and gen merge block
  (void)builder->AddBranch(merge_blk_id);
  new_blocks->push_back(std::move(*new_blk_ptr));
  new_blk_ptr->reset(new BasicBlock(std::move(merge_label)));
  builder->SetInsertPoint(&**new_blk_ptr);
  // Gen phi of new reference and zero, if necessary, and replace the
  // result id of the original reference with that of the Phi. Kill original
  // reference.
  if (new_ref_id != 0) {
    Instruction* phi_inst = builder->AddPhi(
        ref_type_id, {new_ref_id, valid_blk_id, builder->GetNullId(ref_type_id),
                      last_invalid_blk_id});
    context()->ReplaceAllUsesWith(ref_inst_->result_id(),
                                  phi_inst->result_id());
  }
  context()->KillInst(ref_inst_);
}

void InstBindlessCheckPass::GenBoundsCheckCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Look for reference through indexed descriptor. If found, analyze and
  // save components. If not, return.
  if (!AnalyzeDescriptorReference(&*ref_inst_itr)) return;
  if (ptr_inst_->opcode() != SpvOp::SpvOpAccessChain) return;
  // If index and bound both compile-time constants and index < bound,
  // return without changing
  uint32_t var_type_id = var_inst_->type_id();
  Instruction* var_type_inst = get_def_use_mgr()->GetDef(var_type_id);
  uint32_t desc_type_id =
      var_type_inst->GetSingleWordInOperand(kSpvTypePointerTypeIdInIdx);
  Instruction* desc_type_inst = get_def_use_mgr()->GetDef(desc_type_id);
  uint32_t length_id = 0;
  if (desc_type_inst->opcode() == SpvOpTypeArray) {
    length_id =
        desc_type_inst->GetSingleWordInOperand(kSpvTypeArrayLengthIdInIdx);
    Instruction* index_inst = get_def_use_mgr()->GetDef(index_id_);
    Instruction* length_inst = get_def_use_mgr()->GetDef(length_id);
    if (index_inst->opcode() == SpvOpConstant &&
        length_inst->opcode() == SpvOpConstant &&
        index_inst->GetSingleWordInOperand(kSpvConstantValueInIdx) <
            length_inst->GetSingleWordInOperand(kSpvConstantValueInIdx))
      return;
  } else if (!input_length_enabled_ ||
             desc_type_inst->opcode() != SpvOpTypeRuntimeArray) {
    return;
  }
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> new_blk_ptr;
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  uint32_t error_id = builder.GetUintConstantId(kInstErrorBindlessBounds);
  // If length id not yet set, descriptor array is runtime size so
  // generate load of length from stage's debug input buffer.
  if (length_id == 0) {
    assert(desc_type_inst->opcode() == SpvOpTypeRuntimeArray &&
           "unexpected bindless type");
    length_id = GenDebugReadLength(var_id_, &builder);
  }
  // Generate full runtime bounds test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  Instruction* ult_inst =
      builder.AddBinaryOp(GetBoolId(), SpvOpULessThan, index_id_, length_id);
  GenCheckCode(ult_inst->result_id(), error_id, length_id, stage_idx, &builder,
               &new_blk_ptr, new_blocks);
  // Move original block's remaining code into remainder/merge block and add
  // to new blocks
  MovePostludeCode(ref_block_itr, &new_blk_ptr);
  new_blocks->push_back(std::move(new_blk_ptr));
}

void InstBindlessCheckPass::GenInitCheckCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Look for reference through descriptor. If not, return.
  if (!AnalyzeDescriptorReference(&*ref_inst_itr)) return;
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> new_blk_ptr;
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // Read initialization status from debug input buffer. If index id not yet
  // set, binding is single descriptor, so set index to constant 0.
  uint32_t zero_id = builder.GetUintConstantId(0u);
  if (index_id_ == 0) index_id_ = zero_id;
  uint32_t init_id = GenDebugReadInit(var_id_, index_id_, &builder);
  // Generate full runtime non-zero init test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  Instruction* uneq_inst =
      builder.AddBinaryOp(GetBoolId(), SpvOpINotEqual, init_id, zero_id);
  uint32_t error_id = builder.GetUintConstantId(kInstErrorBindlessUninit);
  GenCheckCode(uneq_inst->result_id(), error_id, zero_id, stage_idx, &builder,
               &new_blk_ptr, new_blocks);
  // Move original block's remaining code into remainder/merge block and add
  // to new blocks
  MovePostludeCode(ref_block_itr, &new_blk_ptr);
  new_blocks->push_back(std::move(new_blk_ptr));
}

void InstBindlessCheckPass::InitializeInstBindlessCheck() {
  // Initialize base class
  InitializeInstrument();
  // Look for related extensions
  ext_descriptor_indexing_defined_ = false;
  for (auto& ei : get_module()->extensions()) {
    const char* ext_name =
        reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (strcmp(ext_name, "SPV_EXT_descriptor_indexing") == 0) {
      ext_descriptor_indexing_defined_ = true;
      break;
    }
  }
  // If descriptor indexing extension and runtime array length support enabled,
  // create variable mappings. Length support is always enabled if descriptor
  // init check is enabled.
  if (ext_descriptor_indexing_defined_ && input_length_enabled_)
    for (auto& anno : get_module()->annotations())
      if (anno.opcode() == SpvOpDecorate) {
        if (anno.GetSingleWordInOperand(1u) == SpvDecorationDescriptorSet)
          var2desc_set_[anno.GetSingleWordInOperand(0u)] =
              anno.GetSingleWordInOperand(2u);
        else if (anno.GetSingleWordInOperand(1u) == SpvDecorationBinding)
          var2binding_[anno.GetSingleWordInOperand(0u)] =
              anno.GetSingleWordInOperand(2u);
      }
}

Pass::Status InstBindlessCheckPass::ProcessImpl() {
  // Perform bindless bounds check on each entry point function in module
  InstProcessFunction pfn =
      [this](BasicBlock::iterator ref_inst_itr,
             UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
             std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
        return GenBoundsCheckCode(ref_inst_itr, ref_block_itr, stage_idx,
                                  new_blocks);
      };
  bool modified = InstProcessEntryPointCallTree(pfn);
  if (ext_descriptor_indexing_defined_ && input_init_enabled_) {
    // Perform descriptor initialization check on each entry point function in
    // module
    pfn = [this](BasicBlock::iterator ref_inst_itr,
                 UptrVectorIterator<BasicBlock> ref_block_itr,
                 uint32_t stage_idx,
                 std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
      return GenInitCheckCode(ref_inst_itr, ref_block_itr, stage_idx,
                              new_blocks);
    };
    modified |= InstProcessEntryPointCallTree(pfn);
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status InstBindlessCheckPass::Process() {
  InitializeInstBindlessCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
