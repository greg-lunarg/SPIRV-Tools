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

#include "convert_to_half_pass.h"

#include "source/opt/ir_builder.h"

namespace {

// Indices of operands in SPIR-V instructions
  static const int kEntryPointFunctionIdInIdx = 1;
  static const int kImageSampleDrefIdInIdx = 2;

}  // anonymous namespace

namespace spvtools {
namespace opt {

bool ConvertToHalfPass::is_arithmetic(Instruction* inst) {
  return target_ops_core_.count(inst->opcode()) != 0 ||
      (inst->opcode() == SpvOpExtInst &&
        inst->GetSingleWordInOperand(0) == glsl450_ext_id_ &&
        target_ops_450_.count(inst->GetSingleWordInOperand(1)) != 0);
}

Instruction* ConvertToHalfPass::get_base_type(uint32_t ty_id) {
  Instruction* ty_inst = get_def_use_mgr()->GetDef(ty_id);
  if (ty_inst->opcode() == SpvOpTypeMatrix) {
    uint32_t vty_id = ty_inst->GetSingleWordInOperand(0);
    ty_inst = get_def_use_mgr()->GetDef(vty_id);
  }
  if (ty_inst->opcode() == SpvOpTypeVector) {
    uint32_t cty_id = ty_inst->GetSingleWordInOperand(0);
    ty_inst = get_def_use_mgr()->GetDef(cty_id);
  }
  return ty_inst;
}

bool ConvertToHalfPass::is_float(Instruction* inst, uint32_t width) {
  uint32_t ty_id = inst->type_id();
  if (ty_id == 0) return false;
  Instruction* ty_inst = get_base_type(ty_id);
  if (ty_inst->opcode() != SpvOpTypeFloat)
    return false;
  return ty_inst->GetSingleWordInOperand(0) == width;
}

bool ConvertToHalfPass::is_relaxed(Instruction* inst) {
  if (all_floats_relaxed_) return true;
  uint32_t r_id = inst->result_id();
  for (auto r_inst : get_decoration_mgr()->GetDecorationsFor(r_id, false))
    if (r_inst->opcode() == SpvOpDecorate &&
        r_inst->GetSingleWordInOperand(1) == SpvDecorationRelaxedPrecision)
      return true;
  return false;
}

uint32_t ConvertToHalfPass::get_equiv_float_ty_id(
    uint32_t ty_id, uint32_t width) {
  Instruction* ty_inst = get_def_use_mgr()->GetDef(ty_id);
  // Discover vector count and length
  uint32_t v_cnt = 0;
  uint32_t v_len = 0;
  if (ty_inst->opcode() == SpvOpTypeMatrix) {
    uint32_t vty_id = ty_inst->GetSingleWordInOperand(0);
    v_cnt = ty_inst->GetSingleWordInOperand(1);
    ty_inst = get_def_use_mgr()->GetDef(vty_id);
  }
  if (ty_inst->opcode() == SpvOpTypeVector)
    v_len = ty_inst->GetSingleWordInOperand(1);
  // Build type
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Float float_ty(width);
  analysis::Type* reg_float_ty = type_mgr->GetRegisteredType(&float_ty);
  if (v_len == 0) return type_mgr->GetTypeInstruction(reg_float_ty);
  analysis::Vector vec_ty(reg_float_ty, v_len);
  analysis::Type* reg_vec_ty = type_mgr->GetRegisteredType(&vec_ty);
  if (v_cnt == 0) return type_mgr->GetTypeInstruction(reg_vec_ty);
  analysis::Matrix mat_ty(reg_vec_ty, v_cnt);
  analysis::Type* reg_mat_ty = type_mgr->GetRegisteredType(&mat_ty);
  return type_mgr->GetTypeInstruction(reg_mat_ty);
}

void ConvertToHalfPass::GenConvert(uint32_t ty_id, uint32_t width,
    uint32_t* val_idp, InstructionBuilder* builder) {
  uint32_t nty_id = get_equiv_float_ty_id(ty_id, width);
  Instruction* val_inst = get_def_use_mgr()->GetDef(*val_idp);
  Instruction* cvt_inst;
  if (val_inst->opcode() == SpvOpUndef)
    cvt_inst = builder->AddNullaryOp(nty_id, SpvOpUndef);
  else
    cvt_inst = builder->AddUnaryOp(nty_id, SpvOpFConvert, *val_idp);
  *val_idp = cvt_inst->result_id();
}

bool ConvertToHalfPass::MatConvertCleanup(Instruction* inst) {
  if (inst->opcode() != SpvOpFConvert)
    return false;
  uint32_t mty_id = inst->type_id();
  Instruction* mty_inst = get_def_use_mgr()->GetDef(mty_id);
  if (mty_inst->opcode() != SpvOpTypeMatrix)
    return false;
  uint32_t vty_id = mty_inst->GetSingleWordInOperand(0);
  uint32_t v_cnt = mty_inst->GetSingleWordInOperand(1);
  Instruction* vty_inst = get_def_use_mgr()->GetDef(vty_id);
  uint32_t cty_id = vty_inst->GetSingleWordInOperand(0);
  Instruction* cty_inst = get_def_use_mgr()->GetDef(cty_id);
  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // Create Undef and for each vector in matrix, extract it,
  // convert it and insert it into Undef.
  uint32_t orig_width = (cty_inst->GetSingleWordInOperand(0) == 16) ? 32 : 16;
  uint32_t orig_mat_id = inst->GetSingleWordInOperand(0);
  uint32_t orig_vty_id = get_equiv_float_ty_id(vty_id, orig_width);
  Instruction* mat_inst = builder.AddNullaryOp(mty_id, SpvOpUndef);
  uint32_t mat_id = mat_inst->result_id();
  for (uint32_t vidx = 0; vidx < v_cnt; ++vidx) {
    Instruction* ext_inst = builder.AddIdLiteralOp(orig_vty_id, SpvOpCompositeExtract, orig_mat_id, vidx);
    Instruction* cvt_inst = builder.AddUnaryOp(vty_id, SpvOpFConvert, ext_inst->result_id());
    mat_inst = builder.AddIdIdLiteralOp(mty_id, SpvOpCompositeInsert, cvt_inst->result_id(), mat_id, vidx);
    mat_id = mat_inst->result_id();
  }
  context()->ReplaceAllUsesWith(inst->result_id(), mat_id);
  // Turn original instruction into copy so it is valid.
  inst->SetOpcode(SpvOpCopyObject);
  inst->SetResultType(get_equiv_float_ty_id(mty_id, orig_width));
  get_def_use_mgr()->AnalyzeInstUse(inst);
  return true;
}

void ConvertToHalfPass::RemoveRelaxedDecoration(uint32_t id) {
  context()->get_decoration_mgr()->RemoveDecorationsFrom(
      id, [](const Instruction& dec) {
    if (dec.opcode() == SpvOpDecorate &&
        dec.GetSingleWordInOperand(1u) == SpvDecorationRelaxedPrecision)
      return true;
    else
      return false;
  });
}

bool ConvertToHalfPass::GenHalfCode(Instruction* inst) {
  bool modified = false;
  // Remember id for later deletion of RelaxedPrecision decoration
  bool inst_relaxed = is_relaxed(inst);
  if (inst_relaxed)
    relaxed_ids_.push_back(inst->result_id());
  if (is_arithmetic(inst) && inst_relaxed) {
    // Convert all float operands to half and change type to half
    InstructionBuilder builder(
        context(), inst,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    inst->ForEachInId([&builder,&modified,this](uint32_t* idp) {
      Instruction* op_inst = get_def_use_mgr()->GetDef(*idp);
      if (!is_float(op_inst, 32)) return;
      GenConvert(op_inst->type_id(), 16, idp, &builder);
      modified = true;
    });
    if (modified)
      get_def_use_mgr()->AnalyzeInstUse(inst);
    if (is_float(inst, 32)) {
      inst->SetResultType(get_equiv_float_ty_id(inst->type_id(), 16));
      modified = true;
    }
  }
  else if (inst->opcode() == SpvOpPhi && is_float(inst, 32) && inst_relaxed) {
    // Add converts of operands and change type to half. Converts need to
    // be added to preceeding blocks
    uint32_t ocnt = 0;
    uint32_t* prev_idp;
    inst->ForEachInId([&modified, &ocnt, &prev_idp, this](uint32_t* idp) {
      if (ocnt % 2 == 0) {
        prev_idp = idp;
      }
      else {
        Instruction* val_inst = get_def_use_mgr()->GetDef(*prev_idp);
        if (is_float(val_inst, 32)) {
          BasicBlock* bp = context()->get_instr_block(*idp);
          auto insert_before = bp->tail();
          if (insert_before != bp->begin()) {
            --insert_before;
            if (insert_before->opcode() != SpvOpSelectionMerge &&
              insert_before->opcode() != SpvOpLoopMerge)
              ++insert_before;
          }
          InstructionBuilder builder(
            context(), &*insert_before,
            IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
          GenConvert(val_inst->type_id(), 16, prev_idp, &builder);
          modified = true;
        }
      }
      ++ocnt;
    });
    if (modified)
      get_def_use_mgr()->AnalyzeInstUse(inst);
    inst->SetResultType(get_equiv_float_ty_id(inst->type_id(), 16));
    modified = true;
  } else if (inst->opcode() == SpvOpCompositeExtract && is_float(inst, 32) && inst_relaxed) {
    uint32_t comp_id = inst->GetSingleWordInOperand(0);
    Instruction* comp_inst = get_def_use_mgr()->GetDef(comp_id);
    // If the composite is a relaxed float type, convert it to half
    if (is_float(comp_inst, 32) && is_relaxed(comp_inst)) {
      InstructionBuilder builder(
          context(), inst,
          IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
      GenConvert(comp_inst->type_id(), 16, &comp_id, &builder);
      inst->SetInOperand(0, {comp_id});
      get_def_use_mgr()->AnalyzeInstUse(inst);
      comp_inst = get_def_use_mgr()->GetDef(comp_id);
    }
    // If the composite is a relaxed half type, make sure the type of the
    // extract agrees.
    if (is_float(comp_inst, 16) && !is_float(inst, 16)) {
      inst->SetResultType(get_equiv_float_ty_id(inst->type_id(), 16));
      modified = true;
    }
  } else if (inst->opcode() == SpvOpFConvert) {
    // If operand and result types are the same, change to copy
    uint32_t val_id = inst->GetSingleWordInOperand(0);
    Instruction* val_inst = get_def_use_mgr()->GetDef(val_id);
    if (inst->type_id() == val_inst->type_id()) {
      context()->ReplaceAllUsesWith(inst->result_id(), val_id);
      inst->SetOpcode(SpvOpCopyObject);
      modified = true;
    }
  } else if (sample_ops_.count(inst->opcode()) != 0) {
    // Only need to convert dref args to float32
    if (dref_sample_ops_.count(inst->opcode()) != 0) {
      uint32_t dref_id = inst->GetSingleWordInOperand(kImageSampleDrefIdInIdx);
      Instruction* dref_inst = get_def_use_mgr()->GetDef(dref_id);
      if (is_float(dref_inst, 16) && is_relaxed(dref_inst)) {
        InstructionBuilder builder(
            context(), inst,
            IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
        GenConvert(dref_inst->type_id(), 32, &dref_id, &builder);
        inst->SetInOperand(kImageSampleDrefIdInIdx, { dref_id });
        get_def_use_mgr()->AnalyzeInstUse(inst);
        modified = true;
      }
    }
  } else {
    // If non-relaxed instruction has float16 relaxed operands, need to convert
    // them back to float32
    InstructionBuilder builder(
        context(), inst,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    inst->ForEachInId([&builder, &modified, this](uint32_t* idp) {
      Instruction* op_inst = get_def_use_mgr()->GetDef(*idp);
      if (!is_float(op_inst, 16)) return;
      if (!is_relaxed(op_inst)) return;
      GenConvert(op_inst->type_id(), 32, idp, &builder);
      modified = true;
    });
    if (modified)
      get_def_use_mgr()->AnalyzeInstUse(inst);
  }
  return modified;
}

bool ConvertToHalfPass::ProcessFunction(Function* func) {
  bool modified = false;
  cfg()->ForEachBlockInReversePostOrder(
      func->entry().get(),
      [&modified, this](BasicBlock* bb) {
    for (auto ii = bb->begin(); ii != bb->end(); ++ii)
      modified |= GenHalfCode(&*ii);
  });
  cfg()->ForEachBlockInReversePostOrder(
    func->entry().get(),
    [&modified, this](BasicBlock* bb) {
    for (auto ii = bb->begin(); ii != bb->end(); ++ii)
      modified |= MatConvertCleanup(&*ii);
  });
  return modified;
}

bool ConvertToHalfPass::ProcessCallTreeFromRoots(
    std::queue<uint32_t>* roots) {
  bool modified = false;
  std::unordered_set<uint32_t> done;
  // Process all functions from roots
  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = id2function_.at(fi);
      // Add calls first so we don't add new output function
      context()->AddCalls(fn, roots);
      modified = ProcessFunction(fn) || modified;
    }
  }
  // If modified, make sure module has Float16 capability
  if (modified && !context()->get_feature_mgr()->HasCapability(SpvCapabilityFloat16)) {
    get_module()->AddCapability(MakeUnique<Instruction>(
        context(), SpvOpCapability, 0, 0,
        std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_CAPABILITY, { SpvCapabilityFloat16 }}}));
  }
  // Remove all RelaxedPrecision decorations from instructions and globals
  for (auto c_id : relaxed_ids_)
    RemoveRelaxedDecoration(c_id);
  for (auto& val : get_module()->types_values()) {
    uint32_t v_id = val.result_id();
    if (v_id != 0)
      RemoveRelaxedDecoration(v_id);
  }
  return modified;
}

Pass::Status ConvertToHalfPass::ProcessImpl() {
  // Add together the roots of all entry points
  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  bool modified = ProcessCallTreeFromRoots(&roots);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status ConvertToHalfPass::Process() {
  Initialize();
  return ProcessImpl();
}

void ConvertToHalfPass::Initialize() {
  id2function_.clear();
  for (auto& fn : *get_module()) {
    id2function_[fn.result_id()] = &fn;
  }
  target_ops_core_ = {
    SpvOpVectorExtractDynamic,
    SpvOpVectorInsertDynamic,
    SpvOpVectorShuffle,
    SpvOpCompositeConstruct,
    SpvOpCompositeInsert,
    SpvOpCopyObject,
    SpvOpTranspose,
    SpvOpConvertSToF,
    SpvOpConvertUToF,
    // SpvOpFConvert,
    // SpvOpQuantizeToF16,
    SpvOpFNegate,
    SpvOpFAdd,
    SpvOpFSub,
    SpvOpFMul,
    SpvOpFDiv,
    SpvOpFMod,
    SpvOpVectorTimesScalar,
    SpvOpMatrixTimesScalar,
    SpvOpVectorTimesMatrix,
    SpvOpMatrixTimesVector,
    SpvOpMatrixTimesMatrix,
    SpvOpOuterProduct,
    SpvOpDot,
    SpvOpSelect,
    SpvOpFOrdEqual,
    SpvOpFUnordEqual,
    SpvOpFOrdNotEqual,
    SpvOpFUnordNotEqual,
    SpvOpFOrdLessThan,
    SpvOpFUnordLessThan,
    SpvOpFOrdGreaterThan,
    SpvOpFUnordGreaterThan,
    SpvOpFOrdLessThanEqual,
    SpvOpFUnordLessThanEqual,
    SpvOpFOrdGreaterThanEqual,
    SpvOpFUnordGreaterThanEqual,
  };
  target_ops_450_ = {
    GLSLstd450Round,
    GLSLstd450RoundEven,
    GLSLstd450Trunc,
    GLSLstd450FAbs,
    GLSLstd450FSign,
    GLSLstd450Floor,
    GLSLstd450Ceil,
    GLSLstd450Fract,
    GLSLstd450Radians,
    GLSLstd450Degrees,
    GLSLstd450Sin,
    GLSLstd450Cos,
    GLSLstd450Tan,
    GLSLstd450Asin,
    GLSLstd450Acos,
    GLSLstd450Atan,
    GLSLstd450Sinh,
    GLSLstd450Cosh,
    GLSLstd450Tanh,
    GLSLstd450Asinh,
    GLSLstd450Acosh,
    GLSLstd450Atanh,
    GLSLstd450Atan2,
    GLSLstd450Pow,
    GLSLstd450Exp,
    GLSLstd450Log,
    GLSLstd450Exp2,
    GLSLstd450Log2,
    GLSLstd450Sqrt,
    GLSLstd450InverseSqrt,
    GLSLstd450Determinant,
    GLSLstd450MatrixInverse,
    // TODO(greg-lunarg): GLSLstd450ModfStruct,
    GLSLstd450FMin,
    GLSLstd450FMax,
    GLSLstd450FClamp,
    GLSLstd450FMix,
    GLSLstd450Step,
    GLSLstd450SmoothStep,
    GLSLstd450Fma,
    // TODO(greg-lunarg): GLSLstd450FrexpStruct,
    GLSLstd450Ldexp,
    GLSLstd450Length,
    GLSLstd450Distance,
    GLSLstd450Cross,
    GLSLstd450Normalize,
    GLSLstd450FaceForward,
    GLSLstd450Reflect,
    GLSLstd450Refract,
    GLSLstd450NMin,
    GLSLstd450NMax,
    GLSLstd450NClamp
  };
  sample_ops_ = {
    SpvOpImageSampleImplicitLod,
    SpvOpImageSampleExplicitLod,
    SpvOpImageSampleDrefImplicitLod,
    SpvOpImageSampleDrefExplicitLod,
    SpvOpImageSampleProjImplicitLod,
    SpvOpImageSampleProjExplicitLod,
    SpvOpImageSampleProjDrefImplicitLod,
    SpvOpImageSampleProjDrefExplicitLod,
    SpvOpImageFetch,
    SpvOpImageGather,
    SpvOpImageDrefGather,
    SpvOpImageRead,
    SpvOpImageSparseSampleImplicitLod,
    SpvOpImageSparseSampleExplicitLod,
    SpvOpImageSparseSampleDrefImplicitLod,
    SpvOpImageSparseSampleDrefExplicitLod,
    SpvOpImageSparseSampleProjImplicitLod,
    SpvOpImageSparseSampleProjExplicitLod,
    SpvOpImageSparseSampleProjDrefImplicitLod,
    SpvOpImageSparseSampleProjDrefExplicitLod,
    SpvOpImageSparseFetch,
    SpvOpImageSparseGather,
    SpvOpImageSparseDrefGather,
    SpvOpImageSparseTexelsResident,
    SpvOpImageSparseRead
  };
  dref_sample_ops_ = {
    SpvOpImageSampleDrefImplicitLod,
    SpvOpImageSampleDrefExplicitLod,
    SpvOpImageSampleProjDrefImplicitLod,
    SpvOpImageSampleProjDrefExplicitLod,
    SpvOpImageDrefGather,
    SpvOpImageSparseSampleDrefImplicitLod,
    SpvOpImageSparseSampleDrefExplicitLod,
    SpvOpImageSparseSampleProjDrefImplicitLod,
    SpvOpImageSparseSampleProjDrefExplicitLod,
    SpvOpImageSparseDrefGather,
  };
  // Find GLSL 450 extension id
  glsl450_ext_id_ = 0;
  for (auto& extension : get_module()->ext_inst_imports()) {
    const char* extension_name =
        reinterpret_cast<const char*>(&extension.GetInOperand(0).words[0]);
    if (!strcmp(extension_name, "GLSL.std.450"))
      glsl450_ext_id_ = extension.result_id();
  }
  relaxed_ids_.clear();
}

}  // namespace opt
}  // namespace spvtools
