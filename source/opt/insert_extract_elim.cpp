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

#include <vector>

namespace spvtools {
namespace opt {

namespace {

const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertObjectIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;

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

void InsertExtractElimPass::markInChain(ir::Instruction* insert,
  ir::Instruction* extract) {
  ir::Instruction* inst = insert;          // Capture extract indices
  std::vector<uint32_t> extIndices;
  uint32_t icnt = 0;
  extract->ForEachInOperand([&icnt, &extIndices](const uint32_t* idp) {
    if (icnt > 0)
      extIndices.push_back(*idp);
    ++icnt;
  });
  while (inst->opcode() == SpvOpCompositeInsert) {
    // Once we find a matching insert, we are done
    if (extract != nullptr && ExtInsMatch(extIndices, inst, 0)) {
      liveInserts_.insert(inst->result_id());
      break;
    }
    // If no extract or non-matching intersection, mark live and continue
    if (extract == nullptr || ExtInsConflict(extIndices, inst, 0))
      liveInserts_.insert(inst->result_id());
    // Get next insert in chain
    const uint32_t compId =
        inst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
    inst = get_def_use_mgr()->GetDef(compId);
  }
  // If insert chain ended with phi, do recursive call on each operand
  if (inst->opcode() != SpvOpPhi)
    return;
  if (liveInserts_.find(inst->result_id()) != liveInserts_.end())
    return;
  liveInserts_.insert(inst->result_id());
  icnt = 0;
  inst->ForEachInId([&icnt,&extract,this](uint32_t* idp) {
    if (icnt % 2 == 0) {
      ir::Instruction* pi = get_def_use_mgr()->GetDef(*idp);
      markInChain(pi, extract);
    }
    ++icnt;
  });
  liveInserts_.erase(inst->result_id());
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
      if (ii->opcode() != SpvOpCompositeInsert)
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
            // Mark all inserts in chain that intersect with extract
            markInChain(&*ii, u.inst);
          } break;
          default: {
            // Mark all inserts in chain
            markInChain(&*ii, nullptr);
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
      if (!IsType(ii->type_id(), SpvOpTypeStruct))
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
          uint32_t extOffset = 0;
          uint32_t replId = 0;
          while (cinst->opcode() == SpvOpCompositeInsert) {
            if (ExtInsMatch(extIndices, cinst, extOffset)) {
              // Match! Use inserted value as replacement
              replId = cinst->GetSingleWordInOperand(kInsertObjectIdInIdx);
              break;
            }
            else if (ExtInsConflict(extIndices, cinst, extOffset)) {
              // If extract has fewer indices than the insert, stop searching.
              // Otherwise increment offset of extract indices considered and
              // continue searching through the inserted value
              if (extIndices.size() - extOffset < cinst->NumInOperands() - 2) {
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
            cinst = get_def_use_mgr()->GetDef(cid);
          }
          // If search ended with CompositeConstruct or ConstantComposite
          // and the extract has one index, return the appropriate component.
          // If a vector CompositeConstruct we make sure all preceding
          // components are of component type (not vector composition).
          // TODO(greg-lunarg): Handle multiple-indices, ConstantNull, special
          // vector composition, and additional CompositeInsert.
          if ((cinst->opcode() == SpvOpCompositeConstruct ||
               cinst->opcode() == SpvOpConstantComposite) &&
              (*ii).NumInOperands() - extOffset == 2) {
            uint32_t compIdx = (*ii).GetSingleWordInOperand(extOffset + 1);
            if (IsType(cinst->type_id(), SpvOpTypeVector)) {
              if (compIdx < cinst->NumInOperands()) {
                uint32_t i = 0;
                for (; i <= compIdx; i++) {
                  uint32_t compId = cinst->GetSingleWordInOperand(i);
                  ir::Instruction* compInst = get_def_use_mgr()->GetDef(compId);
                  if (compInst->type_id() != (*ii).type_id())
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

