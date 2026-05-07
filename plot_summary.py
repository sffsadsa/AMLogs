import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.transform import Rotation

# Dữ liệu tổng hợp trong thư mục summary
file_plan = 'summary/planned_path.csv'
file_act_pid = 'summary/actual_path_pid.csv'
file_act_turning = 'summary/actual_path_turning.csv'


def preprocess(df):
    t_val = df['time_sec'].to_numpy() + df['time_nsec'].to_numpy() * 1e-9
    df = df.copy()
    df['t'] = t_val
    df['t_rel'] = df['t'] - t_val[0]
    return {col: df[col].to_numpy() for col in df.columns}


def quaternions_to_euler(data):
    quat = np.column_stack([data['qx'], data['qy'], data['qz'], data['qw']])
    euler = Rotation.from_quat(quat).as_euler('xyz', degrees=False)
    return {
        'roll': euler[:, 0],
        'pitch': euler[:, 1],
        'yaw': euler[:, 2],
    }


def build_interp_plan(df_act, df_plan):
    return {
        'x': np.interp(df_act['t_rel'], df_plan['t_rel'], df_plan['x']),
        'y': np.interp(df_act['t_rel'], df_plan['t_rel'], df_plan['y']),
        'z': np.interp(df_act['t_rel'], df_plan['t_rel'], df_plan['z']),
    }


def build_interp_euler(df_act, df_plan):
    has_quat = all(col in df_act and col in df_plan for col in ['qx', 'qy', 'qz', 'qw'])
    has_euler = all(col in df_act and col in df_plan for col in ['roll_deg', 'pitch_deg', 'yaw_deg'])

    interp_euler = {}
    euler_act = {}
    if has_quat:
        euler_act = quaternions_to_euler(df_act)
        euler_plan = quaternions_to_euler(df_plan)
        for key in ['roll', 'pitch', 'yaw']:
            interp_euler[key] = np.interp(df_act['t_rel'], df_plan['t_rel'], euler_plan[key])
        return interp_euler, euler_act, True

    if has_euler:
        for col in ['roll_deg', 'pitch_deg', 'yaw_deg']:
            key = col.replace('_deg', '')
            interp_euler[key] = np.deg2rad(np.interp(df_act['t_rel'], df_plan['t_rel'], df_plan[col]))
            euler_act[key] = np.deg2rad(df_act[col])
        return interp_euler, euler_act, True

    return {}, {}, False


try:
    df_plan = preprocess(pd.read_csv(file_plan))
    df_pid = preprocess(pd.read_csv(file_act_pid))
    df_turning = preprocess(pd.read_csv(file_act_turning))

    interp_plan_pid = build_interp_plan(df_pid, df_plan)

    interp_euler_pid, euler_pid, has_euler_pid = build_interp_euler(df_pid, df_plan)
    interp_euler_turning, euler_turning, has_euler_turning = build_interp_euler(df_turning, df_plan)
    can_plot_euler = has_euler_pid and has_euler_turning

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    fig.suptitle('So sánh PID và Turning từ thư mục summary', fontsize=15, fontweight='bold')
    grid = plt.GridSpec(3, 2, wspace=0.35, hspace=0.4)

    def fix_axis(ax, title, ylabel):
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

    # --- ĐỒ THỊ THEO THỜI GIAN: VỊ TRÍ ---
    ax_x = fig.add_subplot(grid[0, 0])
    ax_x.plot(df_pid['t_rel'], interp_plan_pid['x'], 'k--', linewidth=1.8, label='Reference')
    ax_x.plot(df_pid['t_rel'], df_pid['x'], color='red', alpha=0.8, label='Actual PID')
    ax_x.plot(df_turning['t_rel'], df_turning['x'], color='blue', alpha=0.8, label='Actual Turning')
    fix_axis(ax_x, 'Tọa độ X', 'X (m)')

    ax_y = fig.add_subplot(grid[1, 0])
    ax_y.plot(df_pid['t_rel'], interp_plan_pid['y'], 'k--', linewidth=1.8, label='Reference')
    ax_y.plot(df_pid['t_rel'], df_pid['y'], color='red', alpha=0.8, label='Actual PID')
    ax_y.plot(df_turning['t_rel'], df_turning['y'], color='blue', alpha=0.8, label='Actual Turning')
    fix_axis(ax_y, 'Tọa độ Y', 'Y (m)')

    ax_z = fig.add_subplot(grid[2, 0])
    ax_z.plot(df_pid['t_rel'], interp_plan_pid['z'], 'k--', linewidth=1.8, label='Reference')
    ax_z.plot(df_pid['t_rel'], df_pid['z'], color='red', alpha=0.8, label='Actual PID')
    ax_z.plot(df_turning['t_rel'], df_turning['z'], color='blue', alpha=0.8, label='Actual Turning')
    ax_z.set_xlabel('Thời gian (s)')
    fix_axis(ax_z, 'Tọa độ Z', 'Z (m)')

    # --- ĐỒ THỊ THEO THỜI GIAN: GÓC EULER ---
    if can_plot_euler:
        euler_specs = [
            ('roll', 'Góc Roll', 'Roll (rad)'),
            ('pitch', 'Góc Pitch', 'Pitch (rad)'),
            ('yaw', 'Góc Yaw', 'Yaw (rad)'),
        ]
        for idx, (key, title, ylabel) in enumerate(euler_specs):
            ax = fig.add_subplot(grid[idx, 1])
            ax.plot(df_pid['t_rel'], interp_euler_pid[key], 'k--', linewidth=1.8, label='Reference')
            ax.plot(df_pid['t_rel'], euler_pid[key], color='red', alpha=0.8, label='Actual PID')
            ax.plot(df_turning['t_rel'], euler_turning[key], color='blue', alpha=0.8, label='Actual Turning')
            if idx == 2:
                ax.set_xlabel('Thời gian (s)')
            fix_axis(ax, title, ylabel)
    else:
        ax_note = fig.add_subplot(grid[:, 1])
        ax_note.axis('off')
        ax_note.text(
            0.5,
            0.5,
            'Khong tim thay du cot quaternion/Euler de ve goc',
            ha='center',
            va='center',
            fontsize=12,
            color='dimgray',
        )

    plt.show()

except Exception:
    import traceback
    print(f"Loi cu the:\n{traceback.format_exc()}")