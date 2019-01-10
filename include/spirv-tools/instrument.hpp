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

#ifndef INCLUDE_SPIRV_TOOLS_INSTRUMENT_HPP_
#define INCLUDE_SPIRV_TOOLS_INSTRUMENT_HPP_

// Shader Instrumentation Interface
//
// This file provides an external interface for applications that wish to
// communicate with shaders instrumented by passes created by:
//
//   CreateInstBindlessCheckPass
//
// More detailed documentation of this routine can be found in optimizer.hpp

namespace spvtools {

// Stream Output Buffer Offsets
//
// The following values provide offsets into the output buffer struct
// generated by InstrumentPass::GenDebugStreamWrite. This method is utilized
// by InstBindlessCheckPass.
//
// The first member of the debug output buffer contains the next available word
// in the data stream to be written. Shaders will atomically read and update
// this value so as not to overwrite each others records. This value must be
// initialized to zero
static const int kDebugOutputSizeOffset = 0;

// The second member of the output buffer is the start of the stream of records
// written by the instrumented shaders. Each record represents a validation
// error. The format of the records is documented below.
static const int kDebugOutputDataOffset = 1;

// Common Stream Record Offsets
//
// The following are offsets to fields which are common to all records written
// to the output stream.
//
// Each record first contains the size of the record in 32-bit words, including
// the size word.
static const int kInstCommonOutSize = 0;

// This is the shader id passed by the layer when the instrumentation pass is
// created.
static const int kInstCommonOutShaderId = 1;

// This is the ordinal position of the instruction within the SPIR-V shader
// which generated the validation error.
static const int kInstCommonOutInstructionIdx = 2;

// This is the stage which generated the validation error. This word is used
// to determine the contents of the next two words in the record.
// 0:Vert, 1:TessCtrl, 2:TessEval, 3:Geom, 4:Frag, 5:Compute
static const int kInstCommonOutStageIdx = 3;
static const int kInstCommonOutCnt = 4;

// Stage-specific Stream Record Offsets
//
// Each stage will contain different values in the next two words of the record
// used to identify which instantiation of the shader generated the validation
// error.
//
// Vertex Shader Output Record Offsets
static const int kInstVertOutVertexId = kInstCommonOutCnt;
static const int kInstVertOutInstanceId = kInstCommonOutCnt + 1;

// Frag Shader Output Record Offsets
static const int kInstFragOutFragCoordX = kInstCommonOutCnt;
static const int kInstFragOutFragCoordY = kInstCommonOutCnt + 1;

// Compute Shader Output Record Offsets
static const int kInstCompOutGlobalInvocationId = kInstCommonOutCnt;
static const int kInstCompOutUnused = kInstCommonOutCnt + 1;

// Tessellation Shader Output Record Offsets
static const int kInstTessOutInvocationId = kInstCommonOutCnt;
static const int kInstTessOutUnused = kInstCommonOutCnt + 1;

// Geometry Shader Output Record Offsets
static const int kInstGeomOutPrimitiveId = kInstCommonOutCnt;
static const int kInstGeomOutInvocationId = kInstCommonOutCnt + 1;

// Size of Common and Stage-specific Members
static const int kInstStageOutCnt = kInstCommonOutCnt + 2;

// Validation Error Code
//
// This identifies the validation error. It also helps to identify
// how many words follow in the record and their meaning.
static const int kInstValidationOutError = kInstStageOutCnt;

// Validation-specific Output Record Offsets
//
// Each different validation will generate a potentially different
// number of words at the end of the record giving more specifics
// about the validation error.
//
// A bindless bounds error will output the index and the bound.
static const int kInstBindlessOutDescIndex = kInstStageOutCnt + 1;
static const int kInstBindlessOutDescBound = kInstStageOutCnt + 2;
static const int kInstBindlessOutCnt = kInstStageOutCnt + 3;

// Maximum Output Record Member Count
static const int kInstMaxOutCnt = kInstStageOutCnt + 3;

// Validation Error Codes
//
// These are the possible validation error codes.
static const int kInstErrorBindlessBounds = 0;

// Direct Input Buffer Offsets
//
// The following values provide member offsets into the input buffers
// consumed by InstrumentPass::GenDebugDirectRead(). This method is utilized
// by InstBindlessCheckPass.
//
// The only object in an input buffer is a runtime array of unsigned
// integers. Each validation will have its own formatting of this array.
static const int kDebugInputDataOffset = 0;

// Debug Buffer Bindings
//
// These are the bindings for the different buffers which are
// read or written by the instrumentation passes.
//
// This is the output buffer written by InstBindlessCheckPass
// and possibly other future validations.
static const int kDebugOutputBindingStream = 0;

// The input buffers for InstBindlessCheckPass start at this
// offset. There is one for each stage and can be indexed using
// SpvExecutionModel_ enums.
static const int kDebugInputBindingOffsetBindless = 1;

}  // namespace spvtools

#endif  // INCLUDE_SPIRV_TOOLS_INSTRUMENT_HPP_
