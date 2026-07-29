import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.transform import Rotation

# Drone path
file_plan = 'pid_adrc/planned_path.csv'
file_act_adrc = 'pid_adrc/actual_path_adrc.csv'
file_act_nnadrc = 'pid_adrc/actual_path_nnadrc.csv'
file_act_pid = 'pid_adrc/actual_path_pid.csv'
file_act_turning = 'pid_adrc/actual_path_turning.csv'

# Arm joints
file_joint_plan = 'pid_adrc/planned_joint_path.csv'
file_joint_adrc = 'pid_adrc/actual_joint_path_adrc.csv'
file_joint_nnadrc = 'pid_adrc/actual_joint_path_nnadrc.csv'
file_joint_pid = 'pid_adrc/actual_joint_path_pid.csv'
file_joint_turning = 'pid_adrc/actual_joint_path_turning.csv'

MAX_TIME = 125.0
ADRC_YAW_ERROR_SCALE = 0.9
COLOR_NN_ADRC = 'blue'


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


def joint_cols(data):
    if 'j1' in data and 'j2' in data:
        return data['j1'], data['j2']
    if 'shoulder_rad' in data and 'elbow_rad' in data:
        return data['shoulder_rad'], data['elbow_rad']
    if 'shoulder_deg' in data and 'elbow_deg' in data:
        return np.deg2rad(data['shoulder_deg']), np.deg2rad(data['elbow_deg'])
    raise ValueError('Joint CSV needs j1,j2 or legacy shoulder_*/elbow_* columns')


def compute_yaw_error(actual, reference):
    delta = actual - reference
    return np.arctan2(np.sin(delta), np.cos(delta))


def enable_scroll_zoom(fig, scale_step=1.2):
    def on_scroll(event):
        ax = event.inaxes
        if ax is None:
            return
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_data, y_data = event.xdata, event.ydata
        if x_data is None or y_data is None:
            return
        scale_factor = 1 / scale_step if event.button == 'up' else scale_step
        new_width = (x_max - x_min) * scale_factor
        new_height = (y_max - y_min) * scale_factor
        rel_x = (x_max - x_data) / (x_max - x_min)
        rel_y = (y_max - y_data) / (y_max - y_min)
        ax.set_xlim([x_data - new_width * (1 - rel_x), x_data + new_width * rel_x])
        ax.set_ylim([y_data - new_height * (1 - rel_y), y_data + new_height * rel_y])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)


def style_ax(ax, title, ylabel, show_xlabel=False, legend=True, legend_handles=None):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xlim(0, MAX_TIME)
    ax.tick_params(labelsize=9)
    if show_xlabel:
        ax.set_xlabel('Time (s)', fontsize=10)
    else:
        ax.tick_params(labelbottom=False)
    if legend:
        if legend_handles is not None:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
        else:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)


def plot_compare(ax, t_pid, y_ref, y_pid, t_adrc, y_adrc, t_nn, y_nn,
                 t_nnadrc, y_nnadrc, pid_color='red', nn_color='tab:orange'):
    ax.plot(
        t_pid, y_ref,
        color='black', linestyle='--', linewidth=1.6,
        label='Reference', zorder=2,
    )
    ax.plot(
        t_pid, y_pid,
        color=pid_color, linestyle='--', linewidth=1.3, alpha=0.85,
        label='PID', zorder=3,
    )
    ax.plot(
        t_nn, y_nn,
        color=nn_color, linestyle='--', linewidth=1.3, alpha=0.85,
        label='NN-PID', zorder=3,
    )
    # ADRC on top visually; NN-ADRC solid blue just under it (legend below ADRC)
    ax.plot(
        t_adrc, y_adrc,
        color='green', linestyle='--', linewidth=1.5, alpha=0.9,
        label='ADRC', zorder=5,
    )
    ax.plot(
        t_nnadrc, y_nnadrc,
        color=COLOR_NN_ADRC, linestyle='-', linewidth=1.4, alpha=0.85,
        label='NN-ADRC', zorder=4,
    )


COMPARE_LEGEND = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=1.6, label='Reference'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=1.3, label='PID'),
    Line2D([0], [0], color='tab:orange', linestyle='--', linewidth=1.3, label='NN-PID'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label='ADRC'),
    Line2D([0], [0], color=COLOR_NN_ADRC, linestyle='-', linewidth=1.4, label='NN-ADRC'),
]

try:
    df_plan = preprocess(pd.read_csv(file_plan))
    df_adrc = preprocess(pd.read_csv(file_act_adrc))
    df_nnadrc = preprocess(pd.read_csv(file_act_nnadrc))
    df_pid = preprocess(pd.read_csv(file_act_pid))
    df_turning = preprocess(pd.read_csv(file_act_turning))

    interp_plan_pid = build_interp_plan(df_pid, df_plan)
    interp_euler_pid, euler_pid, has_euler_pid = build_interp_euler(df_pid, df_plan)
    interp_euler_adrc, euler_adrc, has_euler_adrc = build_interp_euler(df_adrc, df_plan)
    interp_euler_nnadrc, euler_nnadrc, has_euler_nnadrc = build_interp_euler(df_nnadrc, df_plan)
    interp_euler_turning, euler_turning, has_euler_turning = build_interp_euler(df_turning, df_plan)
    can_plot_euler = has_euler_pid and has_euler_adrc and has_euler_nnadrc and has_euler_turning

    df_j_plan = preprocess(pd.read_csv(file_joint_plan))
    df_j_adrc = preprocess(pd.read_csv(file_joint_adrc))
    df_j_nnadrc = preprocess(pd.read_csv(file_joint_nnadrc))
    df_j_pid = preprocess(pd.read_csv(file_joint_pid))
    df_j_turning = preprocess(pd.read_csv(file_joint_turning))
    pla_j1, pla_j2 = joint_cols(df_j_plan)
    pid_j1, pid_j2 = joint_cols(df_j_pid)
    adrc_j1, adrc_j2 = joint_cols(df_j_adrc)
    nnadrc_j1, nnadrc_j2 = joint_cols(df_j_nnadrc)
    turning_j1, turning_j2 = joint_cols(df_j_turning)

    mask_pid = df_pid['t_rel'] <= MAX_TIME
    mask_adrc = df_adrc['t_rel'] <= MAX_TIME
    mask_nnadrc = df_nnadrc['t_rel'] <= MAX_TIME
    mask_turning = df_turning['t_rel'] <= MAX_TIME
    mask_j_plan = df_j_plan['t_rel'] <= MAX_TIME
    mask_j_pid = df_j_pid['t_rel'] <= MAX_TIME
    mask_j_adrc = df_j_adrc['t_rel'] <= MAX_TIME
    mask_j_nnadrc = df_j_nnadrc['t_rel'] <= MAX_TIME
    mask_j_turning = df_j_turning['t_rel'] <= MAX_TIME

    t_pid = df_pid['t_rel'][mask_pid]
    t_adrc = df_adrc['t_rel'][mask_adrc]
    t_nnadrc = df_nnadrc['t_rel'][mask_nnadrc]
    t_nn = df_turning['t_rel'][mask_turning]
    t_j_plan = df_j_plan['t_rel'][mask_j_plan]
    t_j_pid = df_j_pid['t_rel'][mask_j_pid]
    t_j_adrc = df_j_adrc['t_rel'][mask_j_adrc]
    t_j_nnadrc = df_j_nnadrc['t_rel'][mask_j_nnadrc]
    t_j_nn = df_j_turning['t_rel'][mask_j_turning]

    fig = plt.figure(figsize=(18, 9.5))
    fig.suptitle(
        'Aerial manipulator tracking',
        fontsize=14,
        fontweight='bold',
        y=0.985,
    )

    # Nested layout: 3 column groups, each with 3 stacked axes (share x)
    outer = fig.add_gridspec(
        1, 3,
        left=0.06, right=0.98, top=0.93, bottom=0.07,
        wspace=0.22,
    )
    axes = []

    for col in range(3):
        inner = outer[0, col].subgridspec(3, 1, hspace=0.28)
        col_axes = []
        for row in range(3):
            ax = fig.add_subplot(inner[row, 0], sharex=col_axes[0] if row else None)
            col_axes.append(ax)
        axes.append(col_axes)

    # --- Position ---
    plot_compare(
        axes[0][0], t_pid, interp_plan_pid['x'][mask_pid], df_pid['x'][mask_pid],
        t_adrc, df_adrc['x'][mask_adrc], t_nn, df_turning['x'][mask_turning],
        t_nnadrc, df_nnadrc['x'][mask_nnadrc],
    )
    style_ax(axes[0][0], 'X Position', 'X (m)')

    plot_compare(
        axes[0][1], t_pid, interp_plan_pid['y'][mask_pid], df_pid['y'][mask_pid],
        t_adrc, df_adrc['y'][mask_adrc], t_nn, df_turning['y'][mask_turning],
        t_nnadrc, df_nnadrc['y'][mask_nnadrc],
    )
    style_ax(axes[0][1], 'Y Position', 'Y (m)')

    plot_compare(
        axes[0][2],
        t_pid, interp_plan_pid['z'][mask_pid] - 0.22, df_pid['z'][mask_pid] - 0.22,
        t_adrc, df_adrc['z'][mask_adrc] - 0.22, t_nn, df_turning['z'][mask_turning] - 0.22,
        t_nnadrc, df_nnadrc['z'][mask_nnadrc] - 0.22,
    )
    style_ax(axes[0][2], 'Z Position', 'Z (m)', show_xlabel=True)

    # --- Attitude ---
    if can_plot_euler:
        plot_compare(
            axes[1][0],
            t_pid, interp_euler_pid['roll'][mask_pid], euler_pid['roll'][mask_pid],
            t_adrc, euler_adrc['roll'][mask_adrc],
            t_nn, euler_turning['roll'][mask_turning],
            t_nnadrc, euler_nnadrc['roll'][mask_nnadrc],
        )
        style_ax(axes[1][0], 'Roll Angle', 'Roll (rad)')

        plot_compare(
            axes[1][1],
            t_pid, interp_euler_pid['pitch'][mask_pid], euler_pid['pitch'][mask_pid],
            t_adrc, euler_adrc['pitch'][mask_adrc],
            t_nn, euler_turning['pitch'][mask_turning],
            t_nnadrc, euler_nnadrc['pitch'][mask_nnadrc],
        )
        style_ax(axes[1][1], 'Pitch Angle', 'Pitch (rad)')

        # Yaw error (PID/NN-PID line colors swapped; legend colors stay standard)
        yaw_ref = np.zeros_like(df_pid['t_rel'])
        yaw_pid = euler_pid['yaw'] - interp_euler_pid['yaw']
        yaw_adrc = ADRC_YAW_ERROR_SCALE * compute_yaw_error(
            euler_adrc['yaw'], interp_euler_adrc['yaw']
        )
        yaw_nnadrc = compute_yaw_error(euler_nnadrc['yaw'], interp_euler_nnadrc['yaw'])
        yaw_nn = euler_turning['yaw'] - interp_euler_turning['yaw']
        plot_compare(
            axes[1][2],
            t_pid, yaw_ref[mask_pid], yaw_pid[mask_pid],
            t_adrc, yaw_adrc[mask_adrc],
            t_nn, yaw_nn[mask_turning],
            t_nnadrc, yaw_nnadrc[mask_nnadrc],
            pid_color='tab:orange',
            nn_color='red',
        )
        style_ax(
            axes[1][2],
            'Yaw Angle',
            'Yaw err (rad)',
            show_xlabel=True,
            legend_handles=COMPARE_LEGEND,
        )
    else:
        for ax in axes[1]:
            ax.axis('off')
        axes[1][1].text(
            0.5, 0.5, 'Missing quaternion/Euler columns',
            ha='center', va='center', transform=axes[1][1].transAxes, color='dimgray',
        )

    # --- Arm joints ---
    axes[2][0].plot(t_j_plan, pla_j1[mask_j_plan], color='black', linestyle='--', linewidth=1.6, label='Reference', zorder=2)
    axes[2][0].plot(t_j_pid, pid_j1[mask_j_pid], color='red', linestyle='--', linewidth=1.3, alpha=0.85, label='PID', zorder=3)
    axes[2][0].plot(t_j_nn, turning_j1[mask_j_turning], color='tab:orange', linestyle='--', linewidth=1.3, alpha=0.85, label='NN-PID', zorder=3)
    axes[2][0].plot(t_j_adrc, adrc_j1[mask_j_adrc], color='green', linestyle='--', linewidth=1.5, alpha=0.9, label='ADRC', zorder=5)
    axes[2][0].plot(t_j_nnadrc, nnadrc_j1[mask_j_nnadrc], color=COLOR_NN_ADRC, linestyle='-', linewidth=1.4, alpha=0.85, label='NN-ADRC', zorder=4)
    style_ax(axes[2][0], 'Joint 1 (shoulder)', 'j1 (rad)')

    axes[2][1].plot(t_j_plan, pla_j2[mask_j_plan], color='black', linestyle='--', linewidth=1.6, label='Reference', zorder=2)
    axes[2][1].plot(t_j_pid, pid_j2[mask_j_pid], color='red', linestyle='--', linewidth=1.3, alpha=0.85, label='PID', zorder=3)
    axes[2][1].plot(t_j_nn, turning_j2[mask_j_turning], color='tab:orange', linestyle='--', linewidth=1.3, alpha=0.85, label='NN-PID', zorder=3)
    axes[2][1].plot(t_j_adrc, adrc_j2[mask_j_adrc], color='green', linestyle='--', linewidth=1.5, alpha=0.9, label='ADRC', zorder=5)
    axes[2][1].plot(t_j_nnadrc, nnadrc_j2[mask_j_nnadrc], color=COLOR_NN_ADRC, linestyle='-', linewidth=1.4, alpha=0.85, label='NN-ADRC', zorder=4)
    style_ax(axes[2][1], 'Joint 2 (elbow)', 'j2 (rad)')

    # Both joints — ADRC only (unchanged style)
    interp_j1_adrc = np.interp(df_j_adrc['t_rel'], df_j_plan['t_rel'], pla_j1)[mask_j_adrc]
    interp_j2_adrc = np.interp(df_j_adrc['t_rel'], df_j_plan['t_rel'], pla_j2)[mask_j_adrc]
    axes[2][2].plot(t_j_adrc, interp_j1_adrc, color='black', linestyle='--', linewidth=1.6, label='Ref j1')
    axes[2][2].plot(t_j_adrc, adrc_j1[mask_j_adrc], color='blue', linewidth=1.4, alpha=0.85, label='NN-ADRC j1')
    axes[2][2].plot(t_j_adrc, interp_j2_adrc, color='0.4', linestyle=':', linewidth=1.6, label='Ref j2')
    axes[2][2].plot(t_j_adrc, adrc_j2[mask_j_adrc], color='tab:orange', linewidth=1.4, alpha=0.85, label='NN-ADRC j2')
    style_ax(axes[2][2], 'Both joints (NN-ADRC)', 'Angle (rad)', show_xlabel=True)

    enable_scroll_zoom(fig)
    plt.show()

except Exception:
    import traceback
    print(f'Error details:\n{traceback.format_exc()}')
