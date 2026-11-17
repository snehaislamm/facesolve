"""Write the rig and a solved performance as a UsdSkel .usda file.

Uses the standard UsdSkel schemas so DCCs (Maya via mayaUSD, Houdini Solaris,
usdview) can play it back:  SkelRoot > Skeleton > SkelAnimation (blendShapeWeights
time samples) and Mesh (SkelBindingAPI) > BlendShape prims (sparse offsets).
Plain text writer: no pxr dependency, so it runs inside the training container.
"""
from __future__ import annotations

import numpy as np

from .rig import SHAPES


def _vec(a, fmt="{:.4f}"):
    return "[" + ", ".join("(" + ", ".join(fmt.format(c) for c in p) + ")" for p in a) + "]"


def _arr(a, fmt="{}"):
    return "[" + ", ".join(fmt.format(x) for x in a) + "]"


def write_usda(path, rig, weights: np.ndarray, fps: int = 30, tol: float = 1e-3) -> dict:
    T = len(weights)
    I = "[( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )]"
    names = _arr(SHAPES, '"{}"')
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Face"',
        "    metersPerUnit = 0.001",
        '    upAxis = "Y"',
        f"    startTimeCode = 0\n    endTimeCode = {T - 1}\n    timeCodesPerSecond = {fps}",
        ")",
        "",
        'def SkelRoot "Face" (\n    prepend apiSchemas = ["SkelBindingAPI"]\n)\n{',
        '    def Skeleton "Skel" (\n        prepend apiSchemas = ["SkelBindingAPI"]\n    )\n    {',
        '        uniform token[] joints = ["head"]',
        f"        uniform matrix4d[] bindTransforms = {I}",
        f"        uniform matrix4d[] restTransforms = {I}",
        "        rel skel:animationSource = </Face/Skel/Performance>",
        "",
        '        def SkelAnimation "Performance"\n        {',
        '            uniform token[] joints = ["head"]',
        "            float3[] translations = [(0, 0, 0)]",
        "            quatf[] rotations = [(1, 0, 0, 0)]",
        "            half3[] scales = [(1, 1, 1)]",
        f"            uniform token[] blendShapes = {names}",
        "            float[] blendShapeWeights.timeSamples = {",
    ]
    for t, w in enumerate(weights):
        lines.append(f"                {t}: {_arr(w, '{:.4f}')},")
    lines += ["            }", "        }", "    }", ""]
    targets = ", ".join(f"</Face/FaceMesh/{n}>" for n in SHAPES)
    lines += [
        'def Mesh "FaceMesh" (\n        prepend apiSchemas = ["SkelBindingAPI"]\n    )\n    {'.replace("def", "    def", 1),
        f"        int[] faceVertexCounts = {_arr([3] * len(rig.faces))}",
        f"        int[] faceVertexIndices = {_arr(rig.faces.ravel())}",
        f"        point3f[] points = {_vec(rig.neutral)}",
        '        uniform token subdivisionScheme = "none"',
        "        rel skel:skeleton = </Face/Skel>",
        f"        uniform token[] skel:blendShapes = {names}",
        f"        rel skel:blendShapeTargets = [{targets}]",
        '        int[] primvars:skel:jointIndices = [0] (\n            elementSize = 1\n            interpolation = "constant"\n        )',
        '        float[] primvars:skel:jointWeights = [1] (\n            elementSize = 1\n            interpolation = "constant"\n        )',
        "        matrix4d primvars:skel:geomBindTransform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )",
    ]
    n_offsets = 0
    for k, name in enumerate(SHAPES):
        d = rig.deltas[k]
        keep = np.flatnonzero(np.linalg.norm(d, axis=1) > tol)
        n_offsets += len(keep)
        lines += [
            "",
            f'        def BlendShape "{name}"\n        {{',
            f"            uniform vector3f[] offsets = {_vec(d[keep])}",
            f"            uniform int[] pointIndices = {_arr(keep)}",
            "        }",
        ]
    lines += ["    }", "}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return {"frames": T, "blendshapes": len(SHAPES), "sparse_offsets": n_offsets,
            "dense_offsets": len(SHAPES) * rig.V}
