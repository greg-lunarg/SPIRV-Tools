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

#include "relax_float_ops_pass.h"

#include "source/opt/ir_builder.h"

namespace {

// Indices of operands in SPIR-V instructions
  static const int kEntryPointFunctionIdInIdx = 1;
  static const int kImageSampleDrefIdInIdx = 2;

}  // anonymous namespace

namespace spvtools {
namespace opt {

bool RelaxFloatOpsPass::is_relaxable(Instruction* inst) {
  return target_ops_core_.count(inst->opcode()) != 0 ||
      sample_ops_.count(inst->opcode()) != 0 ||
      (inst->opcode() == SpvOpExtInst &&
          inst->GetSingleWordInOperand(0) == glsl450_ext_id_ &&
          target_ops_450_.count(inst->GetSingleWordInOperand(1)) != 0);
}

Instruction* RelaxFloatOpsPass::get_base_type(uint32_t ty_id) {
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

bool RelaxFloatOpsPass::is_float(Instruction* inst, uint32_t width) {
  uint32_t ty_id = inst->type_id();
  if (ty_id == 0) return false;
  Instruction* ty_inst = get_base_type(ty_id);
  if (ty_inst->opcode() != SpvOpTypeFloat)
    return false;
  return ty_inst->GetSingleWordInOperand(0) == width;
}

bool RelaxFloatOpsPass::is_relaxed(uint32_t r_id) {
  for (auto r_inst : get_decoration_mgr()->GetDecorationsFor(r_id, false))
    if (r_inst->GetSingleWordInOperand(0) == SpvOpDecorate)
      return true;
  return false;
}

void RelaxFloatOpsPass::ProcessInst(Instruction* r_inst) {
  uint32_t r_id = r_inst->result_id();
  if (r_id == 0)
    return;
  if (!is_float(r_inst, 32))
    return;
  if (is_relaxed(r_id))
    return;
  if (!is_relaxable(r_inst))
    return;
  ids_to_relax_.insert(r_id);
}

void RelaxFloatOpsPass::ProcessFunction(Function* func) {
  cfg()->ForEachBlockInReversePostOrder(
      func->entry().get(),
      [this](BasicBlock* bb) {
    for (auto ii = bb->begin(); ii != bb->end(); ++ii)
      ProcessInst(&*ii);
  });
}

void RelaxFloatOpsPass::ProcessCallTreeFromRoots(
    std::queue<uint32_t>* roots) {
  std::unordered_set<uint32_t> done;
  // Process all functions from roots
  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = id2function_.at(fi);
      // Add calls first so we don't add new output function
      context()->AddCalls(fn, roots);
      ProcessFunction(fn);
    }
  }
}

Pass::Status RelaxFloatOpsPass::ProcessImpl() {
  // Add together the roots of all entry points
  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  ProcessCallTreeFromRoots(&roots);
  if (ids_to_relax_.empty())
    return Status::SuccessWithoutChange;
  auto deco_mgr = get_decoration_mgr();
  for (auto r_id : ids_to_relax_)
    deco_mgr->AddDecoration(r_id, SpvDecorationRelaxedPrecision);
  return Status::SuccessWithChange;
}

Pass::Status RelaxFloatOpsPass::Process() {
  Initialize();
  return ProcessImpl();
}

void RelaxFloatOpsPass::Initialize() {
  id2function_.clear();
  for (auto& fn : *get_module()) {
    id2function_[fn.result_id()] = &fn;
  }
  target_ops_core_ = {
    SpvOpLoad,
    SpvOpVectorExtractDynamic,
    SpvOpVectorInsertDynamic,
    SpvOpVectorShuffle,
    SpvOpCompositeConstruct,
    SpvOpCompositeInsert,
    SpvOpCopyObject,
    SpvOpTranspose,
    SpvOpConvertSToF,
    SpvOpConvertUToF,
    SpvOpFConvert,
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
  // Find GLSL 450 extension id
  glsl450_ext_id_ = 0;
  for (auto& extension : get_module()->ext_inst_imports()) {
    const char* extension_name =
        reinterpret_cast<const char*>(&extension.GetInOperand(0).words[0]);
    if (!strcmp(extension_name, "GLSL.std.450"))
      glsl450_ext_id_ = extension.result_id();
  }
}

}  // namespace opt
}  // namespace spvtools
