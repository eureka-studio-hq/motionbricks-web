# Third-party notices

## NVIDIA MotionBricks

- **Code** (ported in `web/js/motionbricks.js`, imported by `tools/`): Apache License 2.0,
  Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. https://github.com/NVlabs/motionbricks
- **Model** (`mmm_v6_soma_fp32.onnx`, and the 8-bit build that `tools/build_web_assets.py` makes from it):
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
  Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement. Not included in this repository.
- **Sample motion clips** (the source of the pose library the build script makes):
  NVIDIA Sample Data Evaluation License, for evaluating and testing NVIDIA technologies only.
  Derivative works may not be distributed. Not included in this repository; don't commit or publish
  `web/data/pose_library.json`. The clips are derived from the Bones RigPlay 1 dataset by Bones Studio.

## KayKit: Adventurers Character Pack 1.0

`web/data/characters/*.glb`: Knight, Rogue (hooded), Mage and Barbarian, with animations removed.
Created by Kay Lousberg (https://www.kaylousberg.com). Licensed CC0 1.0
(https://creativecommons.org/publicdomain/zero/1.0/).

## ONNX Runtime Web 1.23.2

`web/vendor/`: MIT License, Copyright (c) Microsoft Corporation.

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## three.js 0.170.0

Loaded at runtime from jsDelivr, not bundled. MIT License, Copyright (c) 2010-2024 three.js authors.
