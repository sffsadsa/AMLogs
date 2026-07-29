"""
Plot manipulator joint trajectories for PID / ADRC / NN-PID comparison.

Layout follows plot_trajectory.plot_joint_path:
  - Column 0: j1, j2 vs time
  - Column 1 top: both joints overlay (time-aligned)
  - Column 1 bottom: joint space (j1–j2)
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = WORKSPACE_DIR / 'pid_adrc'


def preprocess_df(df):
    """Add absolute time column (same as plot_trajectory)."""
    df = df.copy()
    df['t'] = df['time_sec'].values + df['time_nsec'].values * 1e-9
    return df


def load_csv(data_dir, name):
    path = Path(data_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return preprocess_df(pd.read_csv(path))


def _joint_rad_cols(df):
    """Return (j1_rad, j2_rad). Supports new CSV (j1,j2) and legacy names."""
    if 'j1' in df.columns and 'j2' in df.columns:
        return df['j1'].values.astype(float), df['j2'].values.astype(float)
    if 'shoulder_rad' in df.columns and 'elbow_rad' in df.columns:
        return (
            df['shoulder_rad'].values.astype(float),
            df['elbow_rad'].values.astype(float),
        )
    if 'shoulder_deg' in df.columns and 'elbow_deg' in df.columns:
        return (
            np.radians(df['shoulder_deg'].values.astype(float)),
            np.radians(df['elbow_deg'].values.astype(float)),
        )
    raise ValueError(
        'Joint CSV needs j1,j2 (rad) or legacy shoulder_*/elbow_* columns'
    )


def relative_time(df, t_start=None):
    """Convert absolute time to relative. If t_start is None, use series start."""
    t = df['t'].values
    if t_start is None:
        t_start = t[0]
    return t - t_start


def align_pair(raw_pla, raw_act):
    """Align planned/actual on a shared time origin (plot_trajectory style)."""
    pla_j1, pla_j2 = _joint_rad_cols(raw_pla)
    act_j1, act_j2 = _joint_rad_cols(raw_act)

    t_start = min(raw_pla['t'].min(), raw_act['t'].min())
    pla_t = relative_time(raw_pla, t_start)
    act_t = relative_time(raw_act, t_start)

    interp_j1 = np.interp(act_t, pla_t, pla_j1)
    interp_j2 = np.interp(act_t, pla_t, pla_j2)
    return {
        'pla_t': pla_t,
        'act_t': act_t,
        'pla_j1': pla_j1,
        'pla_j2': pla_j2,
        'act_j1': act_j1,
        'act_j2': act_j2,
        'interp_j1': interp_j1,
        'interp_j2': interp_j2,
    }


def fix_axis(ax, title, ylabel):
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')


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


def plot_joint_comparison(data_dir, raw_pla, series):
    """
    series: list of (label, color, aligned_dict)
    Comparison overlay uses each run's own relative time (t - t0 of that run),
    so different controllers can be compared on a common mission timeline.
    """
    # Mission-relative times for overlay comparison
    pla_j1, pla_j2 = _joint_rad_cols(raw_pla)
    pla_t = relative_time(raw_pla)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        f'Arm Joints: {Path(data_dir).name} (PID / ADRC / NN-PID)',
        fontsize=16,
        fontweight='bold',
    )
    grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.35)

    # Column 0: joint vs time
    ax_j1 = fig.add_subplot(grid[0, 0])
    ax_j1.plot(pla_t, pla_j1, 'k--', label='Planned', linewidth=2)
    for label, color, aligned in series:
        ax_j1.plot(
            relative_time(aligned['raw']),
            aligned['j1'],
            color=color,
            alpha=0.7,
            label=label,
        )
    fix_axis(ax_j1, 'Joint 1 (shoulder)', 'Angle (rad)')

    ax_j2 = fig.add_subplot(grid[1, 0])
    ax_j2.plot(pla_t, pla_j2, 'k--', label='Planned', linewidth=2)
    for label, color, aligned in series:
        ax_j2.plot(
            relative_time(aligned['raw']),
            aligned['j2'],
            color=color,
            alpha=0.7,
            label=label,
        )
    ax_j2.set_xlabel('Time (s)')
    fix_axis(ax_j2, 'Joint 2 (elbow)', 'Angle (rad)')

    # Column 1 top: both joints overlay (time-aligned planned→actual for ADRC as reference run)
    # Prefer ADRC if present (same clock as planned); else first series.
    ref_aligned = None
    for label, color, aligned in series:
        if label == 'ADRC':
            ref_aligned = aligned
            break
    if ref_aligned is None:
        ref_aligned = series[0][2]

    pair = align_pair(raw_pla, ref_aligned['raw'])
    ax_cmp = fig.add_subplot(grid[0, 1])
    ax_cmp.plot(pair['act_t'], pair['interp_j1'], 'r--', label='Planned j1', linewidth=2)
    ax_cmp.plot(pair['act_t'], pair['act_j1'], 'b-', label='Actual j1 (ADRC)', alpha=0.7)
    ax_cmp.plot(
        pair['act_t'],
        pair['interp_j2'],
        color='tab:orange',
        linestyle='--',
        label='Planned j2',
        linewidth=2,
    )
    ax_cmp.plot(
        pair['act_t'],
        pair['act_j2'],
        color='tab:green',
        label='Actual j2 (ADRC)',
        alpha=0.7,
    )
    fix_axis(ax_cmp, 'Both joints (time-aligned, ADRC)', 'Angle (rad)')

    # Column 1 bottom: joint space
    ax_phase = fig.add_subplot(grid[1, 1])
    ax_phase.plot(pla_j1, pla_j2, 'k--', label='Planned', linewidth=2)
    ax_phase.scatter([pla_j1[0]], [pla_j2[0]], c='black', s=40, zorder=3)
    for label, color, aligned in series:
        ax_phase.plot(
            aligned['j1'],
            aligned['j2'],
            color=color,
            alpha=0.7,
            label=label,
        )
        ax_phase.scatter(
            [aligned['j1'][0]],
            [aligned['j2'][0]],
            c=color,
            s=40,
            zorder=3,
        )
    ax_phase.set_xlabel('Joint 1 shoulder (rad)')
    ax_phase.set_ylabel('Joint 2 elbow (rad)')
    fix_axis(ax_phase, 'Joint space', 'Joint 2 (rad)')
    ax_phase.set_aspect('equal', adjustable='datalim')

    for label, _, aligned in series:
        t_rel = relative_time(aligned['raw'])
        print(
            f'{label}: samples={len(aligned["j1"])}, duration={t_rel[-1]:.2f}s'
        )
    print(f'Planned: samples={len(pla_j1)}, duration={pla_t[-1]:.2f}s')

    enable_scroll_zoom(fig)
    return fig


def main():
    data_dir = DEFAULT_DATA_DIR
    print(f'Loading joint data from: {data_dir}')

    try:
        raw_pla = load_csv(data_dir, 'planned_joint_path.csv')
        controllers = [
            ('PID', 'red', 'actual_joint_path_pid.csv'),
            ('ADRC', 'blue', 'actual_joint_path_adrc.csv'),
            ('NN-PID', 'green', 'actual_joint_path_turning.csv'),
        ]

        series = []
        for label, color, filename in controllers:
            raw_act = load_csv(data_dir, filename)
            j1, j2 = _joint_rad_cols(raw_act)
            series.append((label, color, {'raw': raw_act, 'j1': j1, 'j2': j2}))

        plot_joint_comparison(data_dir, raw_pla, series)
        plt.show()

    except Exception:
        import traceback
        print('\nError loading or plotting data:')
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
