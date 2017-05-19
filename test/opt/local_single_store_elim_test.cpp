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

#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using LocalSingleStoreElimTest = PassTest<::testing::Test>;


TEST_F(LocalSingleStoreElimTest, EarlyReturnFunctionInlined) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // 
  // float foo(vec4 bar)
  // {
  //     if (bar.x < 0.0)
  //         return 0.0;
  //     return bar.x;
  // }
  // 
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color; 
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_ "foo(vf4;"
OpName %bar "bar"
OpName %color "color"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %float %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%22 = OpLabel
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%23 = OpLoad %v4float %BaseColor
OpStore %param %23
%24 = OpFunctionCall %float %foo_vf4_ %param
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %color %25
%26 = OpLoad %v4float %color
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%false = OpConstantFalse %bool
%main = OpFunction %void None %10
%22 = OpLabel
%35 = OpVariable %_ptr_Function_float Function
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%23 = OpLoad %v4float %BaseColor
OpStore %param %23
OpBranch %36
%36 = OpLabel
OpLoopMerge %37 %38 None
OpBranch %39
%39 = OpLabel
%40 = OpAccessChain %_ptr_Function_float %param %uint_0
%41 = OpLoad %float %40
%42 = OpFOrdLessThan %bool %41 %float_0
OpSelectionMerge %43 None
OpBranchConditional %42 %44 %43
%44 = OpLabel
OpStore %35 %float_0
OpBranch %37
%43 = OpLabel
%45 = OpAccessChain %_ptr_Function_float %param %uint_0
%46 = OpLoad %float %45
OpStore %35 %46
OpBranch %37
%38 = OpLabel
OpBranchConditional %false %36 %37
%37 = OpLabel
%24 = OpLoad %float %35
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %color %25
%26 = OpLoad %v4float %color
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleStoreElimPass>(predefs + before, 
      predefs + after, true, true);
}
TEST_F(LocalSingleStoreElimTest, EarlyReturnInLoopIsNotInlined) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // 
  // float foo(vec4 bar)
  // {
  //     while (true) {
  //         if (bar.x < 0.0)
  //             return 0.0;
  //         return bar.x;
  //     }
  // }
  // 
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color; 
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_ "foo(vf4;"
OpName %bar "bar"
OpName %color "color"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %float %_ptr_Function_v4float
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %10
%23 = OpLabel
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%24 = OpLoad %v4float %BaseColor
OpStore %param %24
%25 = OpFunctionCall %float %foo_vf4_ %param
%26 = OpCompositeConstruct %v4float %25 %25 %25 %25
OpStore %color %26
%27 = OpLoad %v4float %color
OpStore %gl_FragColor %27
OpReturn
OpFunctionEnd
%foo_vf4_ = OpFunction %float None %14
%bar = OpFunctionParameter %_ptr_Function_v4float
%28 = OpLabel
OpBranch %29
%29 = OpLabel
OpLoopMerge %30 %31 None
OpBranch %32
%32 = OpLabel
OpBranchConditional %true %33 %30
%33 = OpLabel
%34 = OpAccessChain %_ptr_Function_float %bar %uint_0
%35 = OpLoad %float %34
%36 = OpFOrdLessThan %bool %35 %float_0
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %37
%38 = OpLabel
OpReturnValue %float_0
%37 = OpLabel
%39 = OpAccessChain %_ptr_Function_float %bar %uint_0
%40 = OpLoad %float %39
OpReturnValue %40
%31 = OpLabel
OpBranch %29
%30 = OpLabel
%41 = OpUndef %float
OpReturnValue %41
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleStoreElimPass>(assembly, assembly, false, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Other types
//    Others?

}  // anonymous namespace
