import sys
import os
import numpy as np

def as_2d(data):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def load_euroc_gt(path):
    # #timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], ...
    data = as_2d(np.genfromtxt(path, delimiter=',', skip_header=1))
    if data.shape[1] < 8:
        raise ValueError(f"Invalid EuRoC ground truth CSV: {path}")
    t = data[:, 0] / 1e9
    p = data[:, 1:4]
    q = data[:, 4:8] # w, x, y, z
    return t, p, q

def load_vio_poses(path, t_bs=None):
    # frame_id,timestamp,qw,qx,qy,qz,tx,ty,tz,vx,vy,vz,ax,ay,az
    data = as_2d(np.genfromtxt(path, delimiter=',', skip_header=1))
    if data.shape[1] < 9:
        raise ValueError(f"Invalid VIO poses CSV: {path}")
    t = data[:, 1]
    q = data[:, 2:6] # w, x, y, z
    p = data[:, 6:9]
    
    if t_bs is not None:
        from scipy.spatial.transform import Rotation as R_sci
        p_body = []
        q_body = []
        
        # EuRoC T_BS maps camera/sensor coordinates into the body frame:
        # p_b = R_BC * p_c + t_BC. The C++ pipeline writes camera pose T_WC,
        # so the comparable body pose is T_WB = T_WC * inv(T_BS).
        t_cb = np.linalg.inv(t_bs)
        r_cb = t_cb[:3, :3]
        p_cb = t_cb[:3, 3]
        
        for i in range(len(p)):
            r_wc = R_sci.from_quat([q[i, 1], q[i, 2], q[i, 3], q[i, 0]]).as_matrix()
            p_wc = p[i]
            
            p_wb = r_wc @ p_cb + p_wc
            r_wb = r_wc @ r_cb
            
            p_body.append(p_wb)
            q_body.append(R_sci.from_matrix(r_wb).as_quat()[[3, 0, 1, 2]]) # w, x, y, z
            
        return t, np.array(p_body), np.array(q_body)
        
    return t, p, q

def load_t_bs(path):
    if not os.path.exists(path):
        return None
    import yaml
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        if 'T_BS' in data:
            return np.array(data['T_BS']['data'], dtype=float).reshape(4, 4)
    return None

def align_umeyama(model, data, estimate_scale=True):
    """Align data trajectory to model using Umeyama SE(3)/Sim(3).
    model: ground truth (N x 3)
    data: estimate (N x 3)
    Returns: aligned data
    """
    mu_m = model.mean(axis=0)
    mu_d = data.mean(axis=0)
    
    m_centered = model - mu_m
    d_centered = data - mu_d
    
    n = model.shape[0]
    
    C = (1.0 / n) * np.dot(d_centered.T, m_centered)
    U, S, Vt = np.linalg.svd(C)

    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1.0

    R = Vt.T @ D @ U.T
        
    s = 1.0
    if estimate_scale:
        var_d = (1.0 / n) * np.sum(np.square(d_centered))
        if var_d > 1e-8:
            s = np.sum(np.diag(D) * S) / var_d

    t = mu_m - s * (R @ mu_d)
    return (s * (R @ data.T)).T + t

def synchronize_by_nearest(t_est, t_gt, max_dt=0.1):
    idx_right = np.searchsorted(t_gt, t_est, side='left')
    indices = np.full(t_est.shape, -1, dtype=int)
    for i, right in enumerate(idx_right):
        candidates = []
        if right < len(t_gt):
            candidates.append(right)
        if right > 0:
            candidates.append(right - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(t_gt[idx] - t_est[i]))
        if abs(t_gt[best] - t_est[i]) <= max_dt:
            indices[i] = best
    return indices

def main():
    gt_path = "data/state_groundtruth_estimate0/data.csv"
    est_path = "results/poses.csv"
    cam_yaml = "data/cam0/sensor-undistorted.yaml"
    
    do_scale = "--scale" in sys.argv
    
    if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        return
    if not os.path.exists(est_path):
        print(f"Estimate not found: {est_path}")
        return
        
    t_bs = None
    try:
        t_bs = load_t_bs(cam_yaml)
        if t_bs is not None:
            print(f"Loaded EuRoC T_BS extrinsics from {cam_yaml}")
    except ImportError:
        print("Warning: 'pyyaml' not found. Skipping extrinsics loading (using Camera frame).")
    except Exception as e:
        print(f"Warning: Failed to load extrinsics: {e}")

    print(f"Loading {gt_path}...")
    t_gt, p_gt, q_gt = load_euroc_gt(gt_path)
    
    print(f"Loading {est_path}...")
    try:
        t_est, p_est, q_est = load_vio_poses(est_path, t_bs)
    except ImportError:
        print("Warning: 'scipy' not found. Using Camera frame for evaluation.")
        t_est, p_est, q_est = load_vio_poses(est_path, None)
    
    print(f"Synchronizing {len(t_est)} estimates with {len(t_gt)} GT poses...")
    
    indices = synchronize_by_nearest(t_est, t_gt, max_dt=0.1)
    valid = indices != -1
    p_est_sync = p_est[valid]
    p_gt_sync = p_gt[indices[valid]]
    
    if len(p_est_sync) < 10:
        print("Not enough synchronized poses!")
        return

    if not np.all(np.isfinite(p_est_sync)):
        print("Estimate contains non-finite positions!")
        return
        
    print(f"Aligned {len(p_est_sync)} poses (Scale estimation: {'ON' if do_scale else 'OFF'}).")
    print(f"Estimate span before alignment: {np.ptp(p_est_sync, axis=0)} m")
    
    # Align trajectories
    p_est_aligned = align_umeyama(p_gt_sync, p_est_sync, estimate_scale=do_scale)
    
    # Calculate ATE
    errors = np.linalg.norm(p_gt_sync - p_est_aligned, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    
    print("\n--- Evaluation Report ---")
    print(f"ATE RMSE:   {rmse:.4f} m")
    print(f"ATE Mean:   {mean_err:.4f} m")
    print(f"ATE Max:    {max_err:.4f} m")
    
    if rmse < 0.1:
        print("Status: EXCELLENT (RMSE < 0.1m)")
    elif rmse < 0.5:
        print("Status: GOOD (RMSE < 0.5m)")
    elif rmse < 1.0:
        print("Status: ACCEPTABLE (RMSE < 1.0m)")
    else:
        print("Status: POOR (RMSE > 1.0m)")

    if "--rerun" in sys.argv:
        import rerun as rr
        rr.init("vio_evaluation", spawn=True)
        
        # Log GT
        rr.log("gt/trajectory", rr.LineStrips3D([p_gt]))
        
        # Log original estimate
        rr.log("est/original", rr.LineStrips3D([p_est]))
        
        # Log aligned estimate
        rr.log("est/aligned", rr.LineStrips3D([p_est_aligned]))
        
        print("Logged trajectories to Rerun.")

if __name__ == "__main__":
    main()
