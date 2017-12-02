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

#include "insert_extract_elim.h"

#include "iterator.h"
#include "ir_context.h"
#include "spirv/1.0/GLSL.std.450.h"

#include <vector>

namespace spvtools {
namespace opt {

namespace {

const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertObjectIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;
const uint32_t kTypeVectorCountInIdx = 1;
const uint32_t kTypeMatrixCountInIdx = 1;
const uint32_t kTypeArrayLengthIdInIdx = 1;
const uint32_t kTypeIntWidthInIdx = 0;
const uint32_t kConstantValueInIdx = 0;
const uint32_t kVectorShuffleVec1IdInIdx = 0;
const uint32_t kVectorShuffleVec2IdInIdx = 1;
const uint32_t kVectorShuffleCompsInIdx = 2;
const uint32_t kTypeVectorCompTypeIdInIdx = 0;
const uint32_t kTypeVectorLengthInIdx = 1;
const uint32_t kExtInstSetIdInIdx = 0;
const uint32_t kExtInstInstructionInIdx = 1;
const uint32_t kFMixXIdInIdx = 1;
const uint32_t kFMixYIdInIdx = 2;
const uint32_t kFMixAIdInIdx = 3;

} // anonymous namespace

bool InsertExtractElimPass::ExtInsMatch(
    const std::vector<uint32_t>& extIndices, const ir::Instruction* insInst,
    const uint32_t extOffset) const {
  uint32_t numIndices = static_cast<uint32_t>(extIndices.size()) - extOffset;
  if (numIndices != insInst->NumInOperands() - 2)
    return false;
  for (uint32_t i = 0; i < numIndices; ++i)
    if (extIndices[i + extOffset] !=
        insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

bool InsertExtractElimPass::ExtInsConflict(
    const std::vector<uint32_t>& extIndices, const ir::Instruction* insInst,
    const uint32_t extOffset) const {
  if (extIndices.size() - extOffset == insInst->NumInOperands() - 2)
    return false;
  uint32_t extNumIndices = static_cast<uint32_t>(extIndices.size()) - extOffset;
  uint32_t insNumIndices = insInst->NumInOperands() - 2;
  uint32_t numIndices = std::min(extNumIndices, insNumIndices);
  for (uint32_t i = 0; i < numIndices; ++i)
    if (extIndices[i + extOffset] !=
        insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

bool InsertExtractElimPass::IsType(uint32_t typeId, SpvOp typeOp) {
  ir::Instruction* typeInst = get_def_use_mgr()->GetDef(typeId);
  return typeInst->opcode() == typeOp;
}

bool InsertExtractElimPass::IsComposite(uint32_t typeId) {
  ir::Instruction* typeInst = get_def_use_mgr()->GetDef(typeId);
  switch (typeInst->opcode()) {
  case SpvOpTypeVector:
  case SpvOpTypeMatrix:
  case SpvOpTypeArray:
  case SpvOpTypeStruct:
    return true;
  default:
    return false;
  }
}

uint32_t InsertExtractElimPass::ComponentNum(uint32_t typeId) {
  ir::Instruction* typeInst = get_def_use_mgr()->GetDef(typeId);
  switch (typeInst->opcode()) {
    case SpvOpTypeVector: {
      return typeInst->GetSingleWordInOperand(kTypeVectorCountInIdx);
    } break;
    case SpvOpTypeMatrix: {
      return typeInst->GetSingleWordInOperand(kTypeMatrixCountInIdx);
    } break;
    case SpvOpTypeArray: {
      uint32_t lenId =
        typeInst->GetSingleWordInOperand(kTypeArrayLengthIdInIdx);
      ir::Instruction* lenInst = get_def_use_mgr()->GetDef(lenId);
      if (lenInst->opcode() != SpvOpConstant)
        return 0;
      uint32_t lenTypeId = lenInst->type_id();
      ir::Instruction* lenTypeInst = get_def_use_mgr()->GetDef(lenTypeId);
      // TODO(greg-lunarg): Support non-32-bit array length
      if (lenTypeInst->GetSingleWordInOperand(kTypeIntWidthInIdx) != 32)
        return 0;
      return lenInst->GetSingleWordInOperand(kConstantValueInIdx);
    } break;
    default: {
      return 0;
    } break;
  }
}

void InsertExtractElimPass::markInsertChain(ir::Instruction* insert,
    std::vector<uint32_t>* pExtIndices, uint32_t extOffset) {
  // If extract indices are empty, mark all subcomponents if type
  // is constant length.
  if (pExtIndices == nullptr) {
    uint32_t cnum = ComponentNum(insert->type_id());
    if (cnum > 0) {
      std::vector<uint32_t> extIndices;
      for (uint32_t i = 0; i < cnum; i++) {
        extIndices.clear();
        extIndices.push_back(i);
        markInsertChain(insert, &extIndices, 0);
      }
      return;
    }
  }
  ir::Instruction* insInst = insert;
  while (insInst->opcode() == SpvOpCompositeInsert) {
    // If no extract indices, mark insert and inserted object and continue
    if (pExtIndices == nullptr) {
      liveInserts_.insert(insInst->result_id());
      uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
      markInsertChain(get_def_use_mgr()->GetDef(objId), nullptr, 0);
    }
    // If extract indices match insert, we are done. Mark insert and
    // inserted object which could also be an insert chain. 
    else if (ExtInsMatch(*pExtIndices, insInst, extOffset)) {
      liveInserts_.insert(insInst->result_id());
      uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
      markInsertChain(get_def_use_mgr()->GetDef(objId), nullptr, 0);
      break;
    }
    // If non-matching intersection, mark insert
    else if (ExtInsConflict(*pExtIndices, insInst, extOffset)) {
      liveInserts_.insert(insInst->result_id());
      // If more extract indices than insert, we are done. Use remaining
      // extract indices to mark inserted object.
      uint32_t numInsertIndices = insInst->NumInOperands() - 2;
      if (pExtIndices->size() - extOffset > numInsertIndices) {
        uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
        markInsertChain(get_def_use_mgr()->GetDef(objId), pExtIndices,
            extOffset + numInsertIndices);
        break;
      }
      // If fewer extract indices than insert, also mark inserted object and
      // continue up chain.
      else {
        uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
        markInsertChain(get_def_use_mgr()->GetDef(objId), nullptr, 0);
      }
    }
    // Get next insert in chain
    const uint32_t compId =
        insInst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
    insInst = get_def_use_mgr()->GetDef(compId);
  }
  // If insert chain ended with phi, do recursive call on each operand
  if (insInst->opcode() != SpvOpPhi)
    return;
  // If phi is already live, we have processed already. Return to
  // avoid infinite loop
  if (liveInserts_.find(insInst->result_id()) != liveInserts_.end())
    return;
  // Insert phi into live set to allow infinite loop check
  liveInserts_.insert(insInst->result_id());
  uint32_t icnt = 0;
  insInst->ForEachInId([&icnt,&pExtIndices,&extOffset,this](uint32_t* idp) {
    if (icnt % 2 == 0) {
      ir::Instruction* pi = get_def_use_mgr()->GetDef(*idp);
      markInsertChain(pi, pExtIndices, extOffset);
    }
    ++icnt;
  });
  // Remove phi from live set when finished
  liveInserts_.erase(insInst->result_id());
}

bool InsertExtractElimPass::EliminateDeadInserts(ir::Function* func) {
  bool modified = false;
  bool lastmodified = true;
  while (lastmodified) {
    lastmodified = EliminateDeadInsertsOnePass(func);
    modified |= lastmodified;
  }
  return modified;
}

bool InsertExtractElimPass::EliminateDeadInsertsOnePass(ir::Function* func) {
  bool modified = false;
  // Mark all live inserts
  liveInserts_.clear();
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      SpvOp op = ii->opcode();
      if (op != SpvOpCompositeInsert && op != SpvOpPhi ||
          !IsComposite(ii->type_id()))
        continue;
      const uint32_t id = ii->result_id();
      const analysis::UseList* uses = get_def_use_mgr()->GetUses(id);
      if (uses == nullptr)
        continue;
      for (const auto u : *uses) {
        const SpvOp op = u.inst->opcode();
        switch (op) {
          case SpvOpCompositeInsert:
          case SpvOpPhi:
            // Use by insert or phi does not cause mark
            break;
          case SpvOpCompositeExtract: {
            // Capture extract indices
            std::vector<uint32_t> extIndices;
            uint32_t icnt = 0;
            u.inst->ForEachInOperand([&icnt, &extIndices]
                (const uint32_t* idp) {
              if (icnt > 0)
                extIndices.push_back(*idp);
              ++icnt;
            });
            // Mark all inserts in chain that intersect with extract
            markInsertChain(&*ii, &extIndices, 0);
          } break;
          default: {
            // Mark all inserts in chain
            markInsertChain(&*ii, nullptr, 0);
          } break;
        }
      }
    }
  }
  // Delete dead inserts
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpCompositeInsert)
        continue;
      const uint32_t id = ii->result_id();
      if (liveInserts_.find(id) != liveInserts_.end())
        continue;
      const uint32_t replId =
          ii->GetSingleWordInOperand(kInsertCompositeIdInIdx);
      (void)get_def_use_mgr()->ReplaceAllUsesWith(id, replId);
      DCEInst(&*ii);
      modified = true;
    }
  }
  return modified;
}

uint32_t InsertExtractElimPass::DoExtract(ir::Instruction* compInst,
    std::vector<uint32_t>* pExtIndices, uint32_t extOffset) {
  ir::Instruction* cinst = compInst;
  uint32_t cid = 0;
  uint32_t replId = 0;
  while (true) {
    if (cinst->opcode() == SpvOpCompositeInsert) {
      if (ExtInsMatch(*pExtIndices, cinst, extOffset)) {
        // Match! Use inserted value as replacement
        replId = cinst->GetSingleWordInOperand(kInsertObjectIdInIdx);
        break;
      }
      else if (ExtInsConflict(*pExtIndices, cinst, extOffset)) {
        // If extract has fewer indices than the insert, stop searching.
        // Otherwise increment offset of extract indices considered and
        // continue searching through the inserted value
        if (pExtIndices->size() - extOffset < cinst->NumInOperands() - 2) {
          break;
        }
        else {
          extOffset += cinst->NumInOperands() - 2;
          cid = cinst->GetSingleWordInOperand(kInsertObjectIdInIdx);
        }
      }
      else {
        // Consider next composite in insert chain
        cid = cinst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
      }
    }
    else if (cinst->opcode() == SpvOpVectorShuffle) {
      // Get length of vector1
      uint32_t v1_id =
        cinst->GetSingleWordInOperand(kVectorShuffleVec1IdInIdx);
      ir::Instruction* v1_inst = get_def_use_mgr()->GetDef(v1_id);
      uint32_t v1_type_id = v1_inst->type_id();
      ir::Instruction* v1_type_inst =
        get_def_use_mgr()->GetDef(v1_type_id);
      uint32_t v1_len =
        v1_type_inst->GetSingleWordInOperand(kTypeVectorLengthInIdx);
      // Get shuffle idx
      uint32_t comp_idx = (*pExtIndices)[extOffset];
      uint32_t shuffle_idx = cinst->GetSingleWordInOperand(
        kVectorShuffleCompsInIdx + comp_idx);
      // If undefined, give up
      // TODO(greg-lunarg): Return OpUndef
      if (shuffle_idx == 0xFFFFFFFF)
        break;
      if (shuffle_idx < v1_len) {
        cid = v1_id;
        (*pExtIndices)[extOffset] = shuffle_idx;
      }
      else {
        cid = cinst->GetSingleWordInOperand(kVectorShuffleVec2IdInIdx);
        (*pExtIndices)[extOffset] = shuffle_idx - v1_len;
      }
    }
    else if (cinst->opcode() == SpvOpExtInst &&
        cinst->GetSingleWordInOperand(kExtInstSetIdInIdx) ==
        get_module()->GetExtInstImportId("GLSL.std.450") &&
        cinst->GetSingleWordInOperand(kExtInstInstructionInIdx) ==
        GLSLstd450FMix) {
      // If mixing value component is 0 or 1 we just match with x or y.
      // Otherwise give up.
      uint32_t comp_idx = (*pExtIndices)[extOffset];
      std::vector<uint32_t> aIndices = {comp_idx};
      uint32_t a_id = cinst->GetSingleWordInOperand(kFMixAIdInIdx);
      ir::Instruction* a_inst = get_def_use_mgr()->GetDef(a_id);
      uint32_t a_comp_id = DoExtract(a_inst, &aIndices, 0);
      if (a_comp_id == 0)
        break;
      ir::Instruction* a_comp_inst = get_def_use_mgr()->GetDef(a_comp_id);
      if (a_comp_inst->opcode() != SpvOpConstant)
        break;
      uint32_t u = a_comp_inst->GetSingleWordInOperand(kConstantValueInIdx);
      float* fp = reinterpret_cast<float*>(&u);
      if (*fp == 0.0)
        cid = cinst->GetSingleWordInOperand(kFMixXIdInIdx);
      else if (*fp == 1.0)
        cid = cinst->GetSingleWordInOperand(kFMixYIdInIdx);
      else
        break;
    }
    else {
      break;
    }
    cinst = get_def_use_mgr()->GetDef(cid);
  }
  // If search ended with CompositeConstruct or ConstantComposite
  // and the extract has one index, return the appropriate component.
  // TODO(greg-lunarg): Handle multiple-indices, ConstantNull, special
  // vector composition, and additional CompositeInsert.
  if (replId == 0 &&
      (cinst->opcode() == SpvOpCompositeConstruct ||
      cinst->opcode() == SpvOpConstantComposite) &&
      (*pExtIndices).size() - extOffset == 1) {
    uint32_t compIdx = (*pExtIndices)[extOffset];
    // If a vector CompositeConstruct we make sure all preceding
    // components are of component type (not vector composition).
    uint32_t ctype_id = cinst->type_id();
    ir::Instruction* ctype_inst = get_def_use_mgr()->GetDef(ctype_id);
    if (ctype_inst->opcode() == SpvOpTypeVector &&
        cinst->opcode() == SpvOpConstantComposite) {
      uint32_t vec_comp_type_id =
          ctype_inst->GetSingleWordInOperand(kTypeVectorCompTypeIdInIdx);
      if (compIdx < cinst->NumInOperands()) {
        uint32_t i = 0;
        for (; i <= compIdx; i++) {
          uint32_t compId = cinst->GetSingleWordInOperand(i);
          ir::Instruction* compInst = get_def_use_mgr()->GetDef(compId);
          if (compInst->type_id() != vec_comp_type_id)
            break;
        }
        if (i > compIdx)
          replId = cinst->GetSingleWordInOperand(compIdx);
      }
    }
    else {
      replId = cinst->GetSingleWordInOperand(compIdx);
    }
  }
  return replId;
}

bool InsertExtractElimPass::EliminateInsertExtract(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
        case SpvOpCompositeExtract: {
          uint32_t cid = ii->GetSingleWordInOperand(kExtractCompositeIdInIdx);
          ir::Instruction* cinst = get_def_use_mgr()->GetDef(cid);
          // Capture extract indices
          std::vector<uint32_t> extIndices;
          uint32_t icnt = 0;
          ii->ForEachInOperand([&icnt,&extIndices](const uint32_t* idp){
            if (icnt > 0)
              extIndices.push_back(*idp);
            ++icnt;
          });
          // Offset of extract indices being compared to insert indices.
          // Offset increases as indices are matched.
          uint32_t replId = DoExtract(cinst, &extIndices, 0);
          if (replId != 0) {
            const uint32_t extId = ii->result_id();
            (void)get_def_use_mgr()->ReplaceAllUsesWith(extId, replId);
            get_def_use_mgr()->KillInst(&*ii);
            modified = true;
          }
        } break;
        default:
          break;
      }
    }
  }
  modified |= EliminateDeadInserts(func);
  return modified;
}

void InsertExtractElimPass::Initialize(ir::IRContext* c) {
  InitializeProcessing(c);

  // Initialize extension whitelist
  InitExtensions();
};

bool InsertExtractElimPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName = reinterpret_cast<const char*>(
        &ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status InsertExtractElimPass::ProcessImpl() {
  // Do not process if any disallowed extensions are enabled
  if (!AllExtensionsSupported())
    return Status::SuccessWithoutChange;
  // Process all entry point functions.
  ProcessFunction pfn = [this](ir::Function* fp) {
    return EliminateInsertExtract(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, get_module());
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InsertExtractElimPass::InsertExtractElimPass() {}

Pass::Status InsertExtractElimPass::Process(ir::IRContext* c) {
  Initialize(c);
  return ProcessImpl();
}

void InsertExtractElimPass::InitExtensions() {
  extensions_whitelist_.clear();
  extensions_whitelist_.insert({
    "SPV_AMD_shader_explicit_vertex_parameter",
    "SPV_AMD_shader_trinary_minmax",
    "SPV_AMD_gcn_shader",
    "SPV_KHR_shader_ballot",
    "SPV_AMD_shader_ballot",
    "SPV_AMD_gpu_shader_half_float",
    "SPV_KHR_shader_draw_parameters",
    "SPV_KHR_subgroup_vote",
    "SPV_KHR_16bit_storage",
    "SPV_KHR_device_group",
    "SPV_KHR_multiview",
    "SPV_NVX_multiview_per_view_attributes",
    "SPV_NV_viewport_array2",
    "SPV_NV_stereo_view_rendering",
    "SPV_NV_sample_mask_override_coverage",
    "SPV_NV_geometry_shader_passthrough",
    "SPV_AMD_texture_gather_bias_lod",
    "SPV_KHR_storage_buffer_storage_class",
    "SPV_KHR_variable_pointers",
    "SPV_AMD_gpu_shader_int16",
    "SPV_KHR_post_depth_coverage",
    "SPV_KHR_shader_atomic_counter_ops",
  });
}

}  // namespace opt
}  // namespace spvtools

