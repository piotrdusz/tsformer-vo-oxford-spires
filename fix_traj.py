import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

if len(sys.argv) != 3:
    print("Usage:")
    print("python flip_orientation.py input.txt output.txt")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        if not line.strip():
            continue

        vals = line.split()

        if len(vals) != 8:
            continue

        ts = vals[0]
        tx, ty, tz = map(float, vals[1:4])
        qx, qy, qz, qw = map(float, vals[4:8])

        # quaternion -> Euler
        r = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = r.as_euler("xyz", degrees=True)

        # Apply transform
        roll = -roll
        pitch = -pitch
        yaw = yaw + 180.0

        # Normalize yaw
        yaw = (yaw + 180.0) % 360.0 - 180.0

        # Euler -> quaternion
        r_new = R.from_euler("xyz", [roll, pitch, yaw], degrees=True)
        qx2, qy2, qz2, qw2 = r_new.as_quat()

        fout.write(
            f"{ts} "
            f"{tx:.9f} {ty:.9f} {tz:.9f} "
            f"{qx2:.9f} {qy2:.9f} {qz2:.9f} {qw2:.9f}\n"
        )

print(f"Saved to {output_file}")
