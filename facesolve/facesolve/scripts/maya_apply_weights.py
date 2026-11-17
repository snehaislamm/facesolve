"""Key solved weights onto a Maya blendShape node.

Run inside Maya's Script Editor (Python tab) after importing
results/solved_take.usda with mayaUSD, or on any mesh whose blendShape node
has targets with the same names as facesolve.rig.SHAPES:

    exec(open("/path/to/facesolve/scripts/maya_apply_weights.py").read())
    apply_weights("/path/to/results/solved_take_weights.csv", "blendShape1")

Untested outside Maya: maya.cmds only exists inside Maya.
"""
import csv

try:
    from maya import cmds
except ImportError:  # outside Maya
    cmds = None


def apply_weights(csv_path, blend_node, start_frame=1):
    if cmds is None:
        raise RuntimeError("run this inside Autodesk Maya")
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    names, frames = rows[0], rows[1:]
    targets = set(cmds.listAttr(blend_node + ".w", multi=True) or [])
    missing = [n for n in names if n not in targets]
    if missing:
        cmds.warning("no target on %s for: %s" % (blend_node, ", ".join(missing)))
    for t, row in enumerate(frames):
        for name, value in zip(names, row):
            if name in targets:
                cmds.setKeyframe(blend_node, attribute=name, time=start_frame + t,
                                 value=float(value))
    cmds.playbackOptions(min=start_frame, max=start_frame + len(frames) - 1)
    return len(frames)
