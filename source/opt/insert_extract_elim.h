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

#ifndef LIBSPIRV_OPT_INSERT_EXTRACT_ELIM_PASS_H_
#define LIBSPIRV_OPT_INSERT_EXTRACT_ELIM_PASS_H_


#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "module.h"
#include "pass.h"
#include "ir_context.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InsertExtractElimPass : public Pass {
 public:
  InsertExtractElimPass();
  const char* name() const override { return "eliminate-insert-extract"; }
  Status Process(ir::IRContext*) override;

 private:
  // Return true if indices of extract |extInst| and insert |insInst| match
  bool ExtInsMatch(
    const ir::Instruction* extInst, const ir::Instruction* insInst) const;

  // Return true if insert |coverer| covers earlier insert |coveree|.
  bool InsCovers(
    const ir::Instruction* coverer, const ir::Instruction* coveree) const;

  // Return true if |insert| is covered by a later insert before use
  // at insert with result |compId|.
  bool IsCoveredInsert(
    const ir::Instruction* insert, const uint32_t compId) const;

  // Return true if indices of extract |extInst| and insert |insInst| conflict,
  // specifically, if the insert changes bits specified by the extract, but
  // changes either more bits or less bits than the extract specifies,
  // meaning the exact value being inserted cannot be used to replace
  // the extract.
  bool ExtInsConflict(
    const ir::Instruction* extInst, const ir::Instruction* insInst) const;

  // Return true if |typeId| is a vector type
  bool IsVectorType(uint32_t typeId);

  // Mark all inserts in chain starting at |ins| that intersect with |ext|.
  // Mark all insert in chain if |ext| is nullptr.
  void markInChain(ir::Instruction* ins, ir::Instruction* ext);

  // Kill all dead inserts in |func|. Replace any reference to the insert
  // with its original composite.
  bool EliminateDeadInserts(ir::Function* func);

  // For all extracts from insert chains in |func|, clone only that part of
  // insert chain that intersects with extract. This frees the larger insert
  // chain to be DCE'd. Return true if |func| is modified.
  bool CloneExtractInsertChains(ir::Function* func);

  // Look for OpExtract on sequence of OpInserts in |func|. If there is an
  // insert with identical indices, replace the extract with the value
  // that is inserted if possible. Specifically, replace if there is no
  // intervening insert which conflicts.
  bool EliminateInsertExtract(ir::Function* func);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  void Initialize(ir::IRContext* c);
  Pass::Status ProcessImpl();

  // Live inserts
  std::unordered_set<uint32_t> liveInserts_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSERT_EXTRACT_ELIM_PASS_H_

