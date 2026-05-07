import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# So sánh trực tiếp dữ liệu từ 2 file plan
file_plan_1 = 'planned_path.csv'
file_plan_2 = 'planned_path_pid.csv'


def load_plan(file_path):
    df = pd.read_csv(file_path).copy()
    required_cols = {'time_sec', 'time_nsec', 'x', 'y', 'z'}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"{file_path} thiếu các cột: {sorted(missing_cols)}")

    df['t'] = df['time_sec'].to_numpy() + df['time_nsec'].to_numpy() * 1e-9
    df['t_rel'] = df['t'] - df['t'].to_numpy()[0]
    return {col: df[col].to_numpy() for col in df.columns}


try:
    df_plan_1 = load_plan(file_plan_1)
    df_plan_2 = load_plan(file_plan_2)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('So sánh dữ liệu từ 2 file plan', fontsize=16, fontweight='bold')
    grid = plt.GridSpec(3, 2, wspace=0.3, hspace=0.4)

    def fix_axis(ax, title, ylabel):
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

    # --- ĐỒ THỊ THEO THỜI GIAN ---
    ax_x = fig.add_subplot(grid[0, 0])
    ax_x.plot(df_plan_1['t_rel'], df_plan_1['x'], 'r--', linewidth=2, label='planned_path.csv')
    ax_x.plot(df_plan_2['t_rel'], df_plan_2['x'], 'b-', alpha=0.8, label='planned_path_pid.csv')
    fix_axis(ax_x, 'Tọa độ X', 'X (m)')

    ax_y = fig.add_subplot(grid[1, 0])
    ax_y.plot(df_plan_1['t_rel'], df_plan_1['y'], 'r--', linewidth=2, label='planned_path.csv')
    ax_y.plot(df_plan_2['t_rel'], df_plan_2['y'], 'g-', alpha=0.8, label='planned_path_pid.csv')
    fix_axis(ax_y, 'Tọa độ Y', 'Y (m)')

    ax_z = fig.add_subplot(grid[2, 0])
    ax_z.plot(df_plan_1['t_rel'], df_plan_1['z'], 'r--', linewidth=2, label='planned_path.csv')
    ax_z.plot(df_plan_2['t_rel'], df_plan_2['z'], 'm-', alpha=0.8, label='planned_path_pid.csv')
    ax_z.set_xlabel('Thời gian tương đối (s)')
    fix_axis(ax_z, 'Tọa độ Z', 'Z (m)')

    # --- ĐỒ THỊ 3D ---
    ax_3d = fig.add_subplot(grid[:, 1], projection='3d')
    ax_3d.plot(df_plan_1['x'], df_plan_1['y'], df_plan_1['z'], 'r--', linewidth=2, label='planned_path.csv')
    ax_3d.plot(df_plan_2['x'], df_plan_2['y'], df_plan_2['z'], 'b-', alpha=0.8, label='planned_path_pid.csv')
    ax_3d.set_title('Quỹ đạo 3D của 2 plan', fontsize=14)
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

except Exception:
    import traceback
    print(f"Lỗi cụ thể:\n{traceback.format_exc()}")