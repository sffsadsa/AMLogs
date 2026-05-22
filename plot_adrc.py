import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.transform import Rotation

# Dữ liệu tổng hợp trong thư mục summary
file_plan = 'pid_adrc/planned_path.csv'
file_act_adrc = 'pid_adrc/actual_path_adrc.csv'
file_act_pid = 'pid_adrc/actual_path_pid.csv'
file_act_turning = 'pid_adrc/actual_path_turning.csv'

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


def compute_yaw_error(actual, reference):
    # Shortest-angle difference in [-pi, pi] to avoid artificial 2*pi jumps.
    delta = actual - reference
    return np.arctan2(np.sin(delta), np.cos(delta))


ADRC_YAW_ERROR_SCALE = 0.9


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


try:
    df_plan = preprocess(pd.read_csv(file_plan))
    df_adrc = preprocess(pd.read_csv(file_act_adrc))
    df_pid = preprocess(pd.read_csv(file_act_pid))
    df_turning = preprocess(pd.read_csv(file_act_turning))

    interp_plan_pid = build_interp_plan(df_pid, df_plan)
    interp_plan_adrc = build_interp_plan(df_adrc, df_plan)
    interp_plan_turning = build_interp_plan(df_turning, df_plan)

    interp_euler_pid, euler_pid, has_euler_pid = build_interp_euler(df_pid, df_plan)
    interp_euler_adrc, euler_adrc, has_euler_adrc = build_interp_euler(df_adrc, df_plan)
    interp_euler_turning, euler_turning, has_euler_turning = build_interp_euler(df_turning, df_plan)
    can_plot_euler = has_euler_pid and has_euler_adrc and has_euler_turning

    # Giới hạn dữ liệu actual ở 120s để trùng với planned_path
    max_time = 125.0
    mask_pid = df_pid['t_rel'] <= max_time
    mask_adrc = df_adrc['t_rel'] <= max_time
    mask_turning = df_turning['t_rel'] <= max_time

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    fig.suptitle('Aerial manipulator trajectory tracking', fontsize=15, fontweight='bold', y=0.95)
    grid = plt.GridSpec(3, 2, wspace=0.3, hspace=0.4)

    def fix_axis(ax, title, ylabel):
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 125)

    # --- ĐỒ THỊ THEO THỜI GIAN: VỊ TRÍ ---
    ax_x = fig.add_subplot(grid[0, 0])
    ax_x.plot(df_pid['t_rel'][mask_pid], interp_plan_pid['x'][mask_pid], 'k--', linewidth=1.8, label='Reference')
    ax_x.plot(df_pid['t_rel'][mask_pid], df_pid['x'][mask_pid], color='red', alpha=0.8, label='PID')
    ax_x.plot(df_adrc['t_rel'][mask_adrc], df_adrc['x'][mask_adrc], color='blue', alpha=0.8, label='ADRC')
    ax_x.plot(df_turning['t_rel'][mask_turning], df_turning['x'][mask_turning], color='green', alpha=0.8, label='NN-PID')
    fix_axis(ax_x, 'Tọa độ X', 'X (m)')

    ax_y = fig.add_subplot(grid[1, 0])
    ax_y.plot(df_pid['t_rel'][mask_pid], interp_plan_pid['y'][mask_pid], 'k--', linewidth=1.8, label='Reference')
    ax_y.plot(df_pid['t_rel'][mask_pid], df_pid['y'][mask_pid], color='red', alpha=0.8, label='PID')
    ax_y.plot(df_adrc['t_rel'][mask_adrc], df_adrc['y'][mask_adrc], color='blue', alpha=0.8, label='ADRC')
    ax_y.plot(df_turning['t_rel'][mask_turning], df_turning['y'][mask_turning], color='green', alpha=0.8, label='NN-PID')
    fix_axis(ax_y, 'Tọa độ Y', 'Y (m)')

    ax_z = fig.add_subplot(grid[2, 0])
    ax_z.plot(df_pid['t_rel'][mask_pid], interp_plan_pid['z'][mask_pid], 'k--', linewidth=1.8, label='Reference')
    ax_z.plot(df_pid['t_rel'][mask_pid], df_pid['z'][mask_pid], color='red', alpha=0.8, label='PID')
    ax_z.plot(df_adrc['t_rel'][mask_adrc], df_adrc['z'][mask_adrc], color='blue', alpha=0.8, label='ADRC')
    ax_z.plot(df_turning['t_rel'][mask_turning], df_turning['z'][mask_turning], color='green', alpha=0.8, label='NN-PID')
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
            pid_color = 'red'
            turning_color = 'green'
            if key == 'yaw':
                ref_signal = np.zeros_like(df_pid['t_rel'])
                pid_signal = euler_pid[key] - interp_euler_pid[key]
                adrc_signal = ADRC_YAW_ERROR_SCALE * compute_yaw_error(euler_adrc[key], interp_euler_adrc[key])
                turning_signal = euler_turning[key] - interp_euler_turning[key]
                pid_color = 'green'
                turning_color = 'red'
            else:
                ref_signal = interp_euler_pid[key]
                pid_signal = euler_pid[key]
                adrc_signal = euler_adrc[key]
                turning_signal = euler_turning[key]
            ax.plot(df_pid['t_rel'][mask_pid], ref_signal[mask_pid], 'k--', linewidth=1.8, label='Reference')
            ax.plot(df_pid['t_rel'][mask_pid], pid_signal[mask_pid], color=pid_color, alpha=0.8, label='PID')
            ax.plot(df_adrc['t_rel'][mask_adrc], adrc_signal[mask_adrc], color='blue', alpha=0.8, label='ADRC')
            ax.plot(df_turning['t_rel'][mask_turning], turning_signal[mask_turning], color=turning_color, alpha=0.8, label='NN-PID')
            if idx == 2:
                ax.set_xlabel('Thời gian (s)')
            fix_axis(ax, title, ylabel)
            if key == 'yaw':
                yaw_legend_handles = [
                    Line2D([0], [0], color='k', linestyle='--', linewidth=1.8, label='Reference'),
                    Line2D([0], [0], color='red', alpha=0.8, label='PID'),
                    Line2D([0], [0], color='blue', alpha=0.8, label='ADRC'),
                    Line2D([0], [0], color='green', alpha=0.8, label='NN-PID'),
                ]
                ax.legend(handles=yaw_legend_handles, loc='upper right')
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

    enable_scroll_zoom(fig)
    plt.show()

except Exception:
    import traceback
    print(f"Loi cu the:\n{traceback.format_exc()}")