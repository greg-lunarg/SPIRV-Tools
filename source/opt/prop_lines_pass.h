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

#ifndef SOURCE_OPT_PROPAGATE_LINES_PASS_H_
#define SOURCE_OPT_PROPAGATE_LINES_PASS_H_

#include "source/opt/function.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class PropagateLinesPass : public Pass {
 using LineProcessFunction = std::function<bool(
     Instruction*, uint32_t*, uint32_t*, uint32_t*)>;

 public:
  const char* name() const override { return "propagate-lines"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisNameMap;
  }

 private:
  // Line propagation is performed on |inst|
  bool PropagateLine(Instruction* inst, uint32_t *file_id, uint32_t *line, 
                     uint32_t *col);

  bool PropagateLinesPass::ProcessLines();

  LineProcessFunction lpfn_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_PROPAGATE_LINES_PASS_H_
