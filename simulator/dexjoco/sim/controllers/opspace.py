from typing import Optional, Tuple, Union

import mujoco
import numpy as np
from dm_robotics.transformations import transformations as tr

# 实现PD控制器
# P = Proportional，比例项：根据“当前位置和目标位置的误差”产生控制量
# D = Derivative，微分项：根据“当前速度”产生阻尼，抑制振荡
# pd_control 用于普通向量，比如位置、关节角
'''
x        当前值，比如当前位置、当前关节角
x_des    目标值，比如目标位置、目标关节角
dx       当前速度，比如线速度、关节速度
kp_kv    PD 增益，通常 shape 是 (维度数, 2)
         第 0 列是 kp，比例增益
         第 1 列是 kv，速度阻尼增益
ddx_max  最大控制输出限制，0 表示不限制
'''
# 控制普通向量，比如位置或关节角
# 输出 = kp * (目标 - 当前) - kv * 当前速度

def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    ddx_max: float = 0.0,
) -> np.ndarray:
    # Compute error.
    x_err = x - x_des
    dx_err = dx

    # Apply gains.
    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]

    # Limit maximum error.
    if ddx_max > 0.0:
        x_err_sq_norm = np.sum(x_err**2)
        ddx_max_sq = ddx_max**2
        if x_err_sq_norm > ddx_max_sq:
            x_err *= ddx_max / np.sqrt(x_err_sq_norm)

    # kp * (x_des - x) - kv * dx
    # 这个返回值通常表示期望加速度、期望速度修正，或者某种任务空间控制量，具体取决于外层控制器怎么用它
    return x_err + dx_err

# pd_control_orientation 用于姿态四元数。
def pd_control_orientation(
    quat: np.ndarray,
    quat_des: np.ndarray,
    w: np.ndarray,
    kp_kv: np.ndarray,
    dw_max: float = 0.0,
) -> np.ndarray:
    # Compute error.
    quat_err = tr.quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = tr.quat_to_axisangle(quat_err)
    w_err = w

    # Apply gains.
    ori_err *= -kp_kv[:, 0]
    w_err *= -kp_kv[:, 1]

    # Limit maximum error.
    if dw_max > 0.0:
        ori_err_sq_norm = np.sum(ori_err**2)
        dw_max_sq = dw_max**2
        if ori_err_sq_norm > dw_max_sq:
            ori_err *= dw_max / np.sqrt(ori_err_sq_norm)

    return ori_err + w_err


# 操作空间控制器，也就是 Operational Space Controller
# 给定机械臂末端目标位置 pos 和目标姿态 ori，计算这几个关节应该施加的力矩 tau，让末端 site 跟踪目标位姿
# policy 或 teleop 给的是末端目标位姿，而 opspace() 把这个末端目标转换成 Panda 7 个关节的力矩
'''
    model     MuJoCo 模型，包含机器人结构、质量、关节、几何体等
    data      MuJoCo 当前仿真状态，包含 qpos、qvel、site_xpos、接触等
    site_id   要控制的末端 site，比如 Panda flange / attachment site
    dof_ids   要控制的关节自由度 id，比如 Panda 7 个关节
    pos       末端目标位置 xyz
    ori       末端目标姿态，可以是四元数，也可以是 3x3 旋转矩阵
    joint     nullspace 目标关节姿态

    后面的参数是控制增益：
    pos_gains              位置 PD 的 kp
    ori_gains              姿态 PD 的 kp
    damping_ratio          阻尼比例，用来自动生成 kd
    nullspace_stiffness    零空间关节姿态保持强度
    max_pos_acceleration   限制最大位置控制量
    max_ori_acceleration   限制最大姿态控制量
    gravity_comp           是否加重力补偿

'''
def opspace(
    model,
    data,
    site_id,
    dof_ids: np.ndarray,
    pos: Optional[np.ndarray] = None,
    ori: Optional[np.ndarray] = None,
    joint: Optional[np.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    ori_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    damping_ratio: float = 1.0,
    nullspace_stiffness: float = 0.5,
    max_pos_acceleration: Optional[float] = None,
    max_ori_acceleration: Optional[float] = None,
    gravity_comp: bool = True,
) -> np.ndarray:
    if pos is None:
        x_des = data.site_xpos[site_id]
    else:
        x_des = np.asarray(pos)
    if ori is None:
        xmat = data.site_xmat[site_id].reshape((3, 3))
        quat_des = tr.mat_to_quat(xmat.reshape((3, 3)))
    else:
        ori = np.asarray(ori)
        if ori.shape == (3, 3):
            quat_des = tr.mat_to_quat(ori)
        else:
            quat_des = ori
    
    # 如果没有传 joint，就以当前关节角为目标；如果传了，比如 _PANDA_HOME
    # 就让机械臂在不影响末端位姿的情况下，尽量靠近 home pose
    if joint is None:
        q_des = data.qpos[dof_ids]
    else:
        q_des = np.asarray(joint)

    # 构造 PD 增益
    kp = np.asarray(pos_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_pos = np.stack([kp, kd], axis=-1)

    kp = np.asarray(ori_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_ori = np.stack([kp, kd], axis=-1)

    # 构造 PD 增益
    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = damping_ratio * 2 * np.sqrt(kp_joint)
    kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

    ddx_max = max_pos_acceleration if max_pos_acceleration is not None else 0.0
    dw_max = max_ori_acceleration if max_ori_acceleration is not None else 0.0


    # 读取当前关节状态
    # Get current state.
    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]

    # Compute Jacobian of the eef site in world frame.
    # 计算末段雅可比矩阵 J
    # J_v   线速度雅可比，描述关节速度如何影响末端线速度
    # J_w   角速度雅可比，描述关节速度如何影响末端角速度
    # 满足：
    # 末端线速度 dx = J_v @ dq
    # 末端角速度 w  = J_w @ dq 

    J_v = np.zeros((3, model.nv), dtype=np.float64)
    J_w = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        model,
        data,
        J_v,
        J_w,
        site_id,
    )
    J_v = J_v[:, dof_ids]
    J_w = J_w[:, dof_ids]
    J = np.concatenate([J_v, J_w], axis=0)

    # Compute position PD control.
    # 计算位置PD控制量
    '''
    x       当前末端位置
    x_des   目标末端位置
    dx      当前末端线速度
    ddx     期望线性加速度/修正量
    '''
    x = data.site_xpos[site_id]
    dx = J_v @ dq
    # 所以如果末端偏离目标，它会产生一个朝目标方向的修正量；如果末端速度太大，会产生阻尼
    ddx = pd_control(
        x=x,
        x_des=x_des,
        dx=dx,
        kp_kv=kp_kv_pos,
        ddx_max=ddx_max,
    )

    # 先计算当前姿态和目标姿态的四元数误差
    # 转成轴角误差
    # 再做 PD 控制
    # Compute orientation PD control.
    quat = tr.mat_to_quat(data.site_xmat[site_id].reshape((3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    w = J_w @ dq
    dw = pd_control_orientation(
        quat=quat,
        quat_des=quat_des,
        w=w,
        kp_kv=kp_kv_ori,
        dw_max=dw_max,
    )

    # Compute inertia matrix in joint space.
    # 计算关节空间惯性矩阵 M
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)
    M = M[dof_ids, :][:, dof_ids]

    # Compute inertia matrix in task space.
    # 把关节空间惯性转换到任务空间
    M_inv = np.linalg.inv(M)
    Mx_inv = J @ M_inv @ J.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    # Compute generalized forces.
    # [期望线性修正 3维, 期望角向修正 3维]
    # 计算主任务关节力矩
    ddx_dw = np.concatenate([ddx, dw], axis=0)
    tau = J.T @ Mx @ ddx_dw

    # Add joint task in nullspace.
    # 这一步计算关节层面的 PD 控制： 
    #   目的不是直接控制末端，而是在不影响末端任务的前提下，让机械臂关节姿态更自然，比如靠近 _PANDA_HOME
    ddq = pd_control(
        x=q,
        x_des=q_des,
        dx=dq,
        kp_kv=kp_kv_joint,
        ddx_max=0.0,
    )
    Jnull = M_inv @ J.T @ Mx
    tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq

    if gravity_comp:
        tau += data.qfrc_bias[dof_ids]
    return tau
'''
整体的控制流程：

    目标末端位置 pos、目标末端姿态 ori
            ↓
    计算当前末端位置/姿态误差
            ↓
    位置 PD + 姿态 PD
            ↓
    得到任务空间 6D 修正量 [ddx, dw]
            ↓
    用雅可比 J 和任务空间惯性 Mx 转成关节力矩 tau
            ↓
    加零空间关节姿态保持
            ↓
    加重力补偿
            ↓
    返回 7 维关节力矩

和普通 IK 的区别：
    这个函数不是单纯求逆运动学 IK。
IK 通常是：
    目标末端位姿 → 目标关节角 q_des
而这里是操作空间控制：
    目标末端位姿 → 当前应施加的关节力矩 tau
它直接在动力学层面控制机械臂，所以会考虑：
    当前速度
    惯性矩阵
    雅可比
    重力/科氏项
    零空间姿态
'''
