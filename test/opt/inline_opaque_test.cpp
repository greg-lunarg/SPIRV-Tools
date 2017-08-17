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

using InlineOpaqueTest = PassTest<::testing::Test>;

TEST_F(InlineOpaqueTest, InlineOpaqueArg) {
  // Function with opaque argument is inlined.
  // TODO(greg-lunarg): Add HLSL code

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %outColor %texCoords
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpMemberName %S_t 2 "smp"
OpName %foo_struct_S_t_vf2_vf21_ "foo(struct-S_t-vf2-vf21;"
OpName %s "s"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %s0 "s0"
OpName %texCoords "texCoords"
OpName %param "param"
OpDecorate %sampler15 DescriptorSet 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%17 = OpTypeImage %float 2D 0 0 0 1 Unknown
%18 = OpTypeSampledImage %17
%S_t = OpTypeStruct %v2float %v2float %18
%_ptr_Function_S_t = OpTypePointer Function %S_t
%20 = OpTypeFunction %void %_ptr_Function_S_t
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%_ptr_Function_18 = OpTypePointer Function %18
%sampler15 = OpVariable %_ptr_UniformConstant_18 UniformConstant
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %12
%28 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function 
%param = OpVariable %_ptr_Function_S_t Function
%29 = OpLoad %v2float %texCoords
%30 = OpAccessChain %_ptr_Function_v2float %s0 %int_0
OpStore %30 %29
%31 = OpLoad %18 %sampler15
%32 = OpAccessChain %_ptr_Function_18 %s0 %int_2
OpStore %32 %31
%33 = OpLoad %S_t %s0 
OpStore %param %33
%34 = OpFunctionCall %void %foo_struct_S_t_vf2_vf21_ %param
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %12
%28 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%param = OpVariable %_ptr_Function_S_t Function
%29 = OpLoad %v2float %texCoords
%30 = OpAccessChain %_ptr_Function_v2float %s0 %int_0
OpStore %30 %29
%31 = OpLoad %18 %sampler15
%32 = OpAccessChain %_ptr_Function_18 %s0 %int_2
OpStore %32 %31
%33 = OpLoad %S_t %s0
OpStore %param %33
%41 = OpAccessChain %_ptr_Function_18 %param %int_2
%42 = OpLoad %18 %41
%43 = OpAccessChain %_ptr_Function_v2float %param %int_0
%44 = OpLoad %v2float %43
%45 = OpImageSampleImplicitLod %v4float %42 %44
OpStore %outColor %45
OpReturn
OpFunctionEnd
)";

  const std::string remains =
      R"(%foo_struct_S_t_vf2_vf21_ = OpFunction %void None %20
%s = OpFunctionParameter %_ptr_Function_S_t
%35 = OpLabel
%36 = OpAccessChain %_ptr_Function_18 %s %int_2
%37 = OpLoad %18 %36
%38 = OpAccessChain %_ptr_Function_v2float %s %int_0
%39 = OpLoad %v2float %38
%40 = OpImageSampleImplicitLod %v4float %37 %39
OpStore %outColor %40
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InlineOpaquePass>(
      predefs + before + remains,
      predefs + after + remains, true, true);
}

TEST_F(InlineOpaqueTest, InlineOpaqueReturn) {
  // Function with opaque return value is inlined.
  // TODO(greg-lunarg): Add HLSL code

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %texCoords %outColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_ "foo("
OpName %texCoords "texCoords"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %sampler16 "sampler16"
OpDecorate %sampler15 DescriptorSet 0
OpDecorate %sampler16 DescriptorSet 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%bool = OpTypeBool
%false = OpConstantFalse %bool
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
%float_0 = OpConstant %float 0
%16 = OpConstantComposite %v2float %float_0 %float_0
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%19 = OpTypeImage %float 2D 0 0 0 1 Unknown
%20 = OpTypeSampledImage %19
%21 = OpTypeFunction %20
%_ptr_UniformConstant_20 = OpTypePointer UniformConstant %20
%_ptr_Function_20 = OpTypePointer Function %20
%sampler15 = OpVariable %_ptr_UniformConstant_20 UniformConstant
%sampler16 = OpVariable %_ptr_UniformConstant_20 UniformConstant
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%24 = OpLabel
%25 = OpVariable %_ptr_Function_20 Function 
%26 = OpFunctionCall %20 %foo_
OpStore %25 %26
%27 = OpLoad %20 %25
%28 = OpLoad %v2float %texCoords
%29 = OpImageSampleImplicitLod %v4float %27 %28
OpStore %outColor %29
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%24 = OpLabel
%34 = OpVariable %_ptr_Function_20 Function
%35 = OpVariable %_ptr_Function_20 Function
%25 = OpVariable %_ptr_Function_20 Function
%36 = OpLoad %20 %sampler16
OpStore %34 %36
%37 = OpLoad %20 %34
OpStore %35 %37
%26 = OpLoad %20 %35
OpStore %25 %26
%27 = OpLoad %20 %25
%28 = OpLoad %v2float %texCoords
%29 = OpImageSampleImplicitLod %v4float %27 %28
OpStore %outColor %29
OpReturn
OpFunctionEnd
)";

  const std::string remains =
      R"(%foo_ = OpFunction %20 None %21
%30 = OpLabel
%31 = OpVariable %_ptr_Function_20 Function
%32 = OpLoad %20 %sampler16
OpStore %31 %32
%33 = OpLoad %20 %31
OpReturnValue %33
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InlineOpaquePass>(
      predefs + before + remains,
      predefs + after + remains, true, true);
}

}  // anonymous namespace
