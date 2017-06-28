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

using LocalSSAElimTest = PassTest<::testing::Test>;

TEST_F(LocalSSAElimTest, ForLoop) {
  // #version 140
  // 
  // in vec4 BC;
  // out float fo;
  // 
  // void main()
  // {
  //     float f = 0.0;
  //     for (int i=0; i<4; i++) {
  //       f = f + BC[i];
  //     }
  //     fo = f;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BC %fo
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %f "f"
OpName %i "i"
OpName %BC "BC"
OpName %fo "fo"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_4 = OpConstant %int 4
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BC = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%fo = OpVariable %_ptr_Output_float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %f %float_0
OpStore %i %int_0
OpBranch %23
%23 = OpLabel
OpLoopMerge %24 %25 None
OpBranch %26
%26 = OpLabel
%27 = OpLoad %int %i
%28 = OpSLessThan %bool %27 %int_4
OpBranchConditional %28 %29 %24
%29 = OpLabel
%30 = OpLoad %float %f
%31 = OpLoad %int %i
%32 = OpAccessChain %_ptr_Input_float %BC %31
%33 = OpLoad %float %32
%34 = OpFAdd %float %30 %33
OpStore %f %34
OpBranch %25
%25 = OpLabel
%35 = OpLoad %int %i
%36 = OpIAdd %int %35 %int_1
OpStore %i %36
OpBranch %23
%24 = OpLabel
%37 = OpLoad %float %f
OpStore %fo %37
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpBranch %23
%23 = OpLabel
%38 = OpPhi %float %float_0 %22 %34 %25
%39 = OpPhi %int %int_0 %22 %36 %25
OpLoopMerge %24 %25 None
OpBranch %26
%26 = OpLabel
%28 = OpSLessThan %bool %39 %int_4
OpBranchConditional %28 %29 %24
%29 = OpLabel
%32 = OpAccessChain %_ptr_Input_float %BC %39
%33 = OpLoad %float %32
%34 = OpFAdd %float %38 %33
OpBranch %25
%25 = OpLabel
%36 = OpIAdd %int %39 %int_1
OpBranch %23
%24 = OpLabel
OpStore %fo %38
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSSAElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(LocalSSAElimTest, IfThenElse) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // in float f;
  // 
  // void main()
  // {
  //     vec4 v;
  //     if (f >= 0)
  //       v = BaseColor * 0.5;
  //     else
  //       v = BaseColor + vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %f %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %f "f"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
%f = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%float_0_5 = OpConstant %float 0.5
%float_1 = OpConstant %float 1
%18 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %float %f
%22 = OpFOrdGreaterThanEqual %bool %21 %float_0
OpSelectionMerge %23 None
OpBranchConditional %22 %24 %25
%24 = OpLabel
%26 = OpLoad %v4float %BaseColor
%27 = OpVectorTimesScalar %v4float %26 %float_0_5
OpStore %v %27
OpBranch %23
%25 = OpLabel
%28 = OpLoad %v4float %BaseColor
%29 = OpFAdd %v4float %28 %18
OpStore %v %29
OpBranch %23
%23 = OpLabel
%30 = OpLoad %v4float %v
OpStore %gl_FragColor %30
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %float %f
%22 = OpFOrdGreaterThanEqual %bool %21 %float_0
OpSelectionMerge %23 None
OpBranchConditional %22 %24 %25
%24 = OpLabel
%26 = OpLoad %v4float %BaseColor
%27 = OpVectorTimesScalar %v4float %26 %float_0_5
OpBranch %23
%25 = OpLabel
%28 = OpLoad %v4float %BaseColor
%29 = OpFAdd %v4float %28 %18
OpBranch %23
%23 = OpLabel
%31 = OpPhi %v4float %27 %24 %29 %25
OpStore %gl_FragColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSSAElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(LocalSSAElimTest, IfThen) {

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %f %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%f = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%float_0_5 = OpConstant %float 0.5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %v %19
%20 = OpLoad %float %f
%21 = OpFOrdLessThanEqual %bool %20 %float_0
OpSelectionMerge %22 None
OpBranchConditional %21 %23 %22
%23 = OpLabel
%24 = OpLoad %v4float %v
%25 = OpVectorTimesScalar %v4float %24 %float_0_5
OpStore %v %25
OpBranch %22
%22 = OpLabel
%26 = OpLoad %v4float %v
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %float %f
%21 = OpFOrdLessThanEqual %bool %20 %float_0
OpSelectionMerge %22 None
OpBranchConditional %21 %23 %22
%23 = OpLabel
%25 = OpVectorTimesScalar %v4float %19 %float_0_5
OpBranch %22
%22 = OpLabel
%27 = OpPhi %v4float %19 %18 %25 %23
OpStore %gl_FragColor %27
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSSAElimPass>(
      predefs + before, predefs + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    No optimization in the presence of
//      access chains
//      function calls
//    Others?

}  // anonymous namespace
