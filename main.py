from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R
from models.lie_encoder import predict_omega

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrajectoryInput(BaseModel):
    points: List[List[float]]  # list of [x, y, z]

def so3_log_map(points):
    result = []
    for xyz in points:
        try:
            xyz = np.array(xyz, dtype=float)
            if xyz.shape != (3,):
                continue
            norm = np.linalg.norm(xyz)
            if norm == 0:
                result.append([0.0, 0.0, 0.0])
            else:
                r = R.from_rotvec(xyz)
                logvec = r.as_rotvec() * 0.7  # simulate compression
                result.append(logvec.tolist())
        except Exception as e:
            print(f"Skipping invalid entry {xyz}: {e}")
    return np.array(result)

@app.post("/transform")
def transform_trajectory(data: TrajectoryInput):
    arr = np.array(data.points)
    if arr.shape[0] < 2:
        return {"error": "Need at least two points to compute trajectory"}

    xi_seq = []
    for i in range(1, len(arr)):
        delta = arr[i] - arr[i - 1]  # placeholder for relative motion
        xi_seq.append(delta.tolist())

    if len(xi_seq) != 99:  # pad or truncate for SEQ_LEN = 100
        xi_seq = (xi_seq + [[0, 0, 0]] * 99)[:99]

    pred_omega = predict_omega(xi_seq)
    return {"omega": pred_omega}