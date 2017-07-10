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

using BlockMergeTest = PassTest<::testing::Test>;

TEST_F(BlockMergeTest, Basic1) {
  // Note: This test exemplifies the following:
  // - Common uniform (%_) load floated to nearest non-controlled block
  // - Common extract (g_F) floated to non-controlled block
  // - Non-common extract (g_F2) not floated, but common uniform load shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  // 
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  //     float g_F2;
  // } ;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (fi > 0) {
  //       v = v * g_F;
  //     }
  //     else {
  //       float f2 = g_F2 - g_F;
  //       v = v * f2;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %f2 "f2"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_float = OpTypePointer Function %float
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%35 = OpLoad %float %34
%36 = OpVectorTimesScalar %v4float %33 %35
OpStore %v %36
OpBranch %30
%32 = OpLabel
%37 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%38 = OpLoad %float %37
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39
%41 = OpFSub %float %38 %40
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%52 = OpLoad %U_t %_
%53 = OpCompositeExtract %float %52 0
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%36 = OpVectorTimesScalar %v4float %33 %53
OpStore %v %36
OpBranch %30
%32 = OpLabel
%49 = OpCompositeExtract %float %52 1
%41 = OpFSub %float %49 %53
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CommonUniformElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(BlockMergeTest, Basic2) {
  // Note: This test exemplifies the following:
  // - Common uniform (%_) load floated to nearest non-controlled block
  // - Common extract (g_F) floated to non-controlled block
  // - Non-common extract (g_F2) not floated, but common uniform load shared
  //

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %fi %fi2 %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %f "f"
OpName %fi "fi"
OpName %fi2 "fi2"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%fi2 = OpVariable %_ptr_Input_float Input
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%26 = OpLoad %float %fi
OpStore %f %26
%27 = OpLoad %float %f
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %float %f
%32 = OpFNegate %float %31
OpStore %f %32
OpBranch %29
%29 = OpLabel
%33 = OpLoad %float %fi2
%34 = OpFOrdGreaterThan %bool %33 %float_0
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %37
%36 = OpLabel
%38 = OpLoad %float %f
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39
%41 = OpFMul %float %38 %40
OpStore %f %41
OpBranch %35
%37 = OpLabel
%42 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%43 = OpLoad %float %42
%44 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%45 = OpLoad %float %44
%46 = OpFSub %float %43 %45
OpStore %f %46
OpBranch %35
%35 = OpLabel
%47 = OpLoad %v4float %BaseColor
%48 = OpLoad %float %f
%49 = OpVectorTimesScalar %v4float %47 %48
OpStore %gl_FragColor %49
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%26 = OpLoad %float %fi
OpStore %f %26
%27 = OpLoad %float %f
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %float %f
%32 = OpFNegate %float %31
OpStore %f %32
OpBranch %29
%29 = OpLabel
%56 = OpLoad %U_t %_
%57 = OpCompositeExtract %float %56 0
%33 = OpLoad %float %fi2
%34 = OpFOrdGreaterThan %bool %33 %float_0
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %37
%36 = OpLabel
%38 = OpLoad %float %f
%41 = OpFMul %float %38 %57
OpStore %f %41
OpBranch %35
%37 = OpLabel
%53 = OpCompositeExtract %float %56 1
%46 = OpFSub %float %53 %57
OpStore %f %46
OpBranch %35
%35 = OpLabel
%47 = OpLoad %v4float %BaseColor
%48 = OpLoad %float %f
%49 = OpVectorTimesScalar %v4float %47 %48
OpStore %gl_FragColor %49
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CommonUniformElimPass>(
      predefs + before, predefs + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Disqualifying cases: extensions, decorations, non-logical addressing,
//      non-structured control flow
//    Others?

}  // anonymous namespace
