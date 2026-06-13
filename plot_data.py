import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.transform import Rotation

# 1. Khai báo tên file
file_act = '1306/actual_path_20260613_145816.csv'
file_pla = '1306/planned_path_20260613_145816.csv'

try:
    # 2. Đọc và xử lý thời gian (Ép về NumPy array để tránh lỗi indexing)
    def preprocess(df):
        # Tạo cột t trước
        t_val = df['time_sec'].values + df['time_nsec'].values * 1e-9
        df['t'] = t_val
        return df

    def quaternions_to_euler(df):
        quat = np.column_stack([df['qx'], df['qy'], df['qz'], df['qw']])
        euler = Rotation.from_quat(quat).as_euler('xyz', degrees=False)
        return {
            'roll': euler[:, 0],
            'pitch': euler[:, 1],
            'yaw': euler[:, 2],
        }

    # Đọc file và chuyển đổi toàn bộ DataFrame sang Dictionary của Numpy Arrays
    # Cách này giúp các hàm như np.interp không bao giờ bị lỗi indexing nữa
    raw_act = preprocess(pd.read_csv(file_act))
    raw_pla = preprocess(pd.read_csv(file_pla))
    
    # Ép kiểu dữ liệu về mảng NumPy thuần túy
    df_act = {col: raw_act[col].values for col in raw_act.columns}
    df_pla = {col: raw_pla[col].values for col in raw_pla.columns}

    # Đưa thời gian về gốc 0
    t_start = df_act['t'].min()
    df_act['t_rel'] = df_act['t'] - t_start
    df_pla['t_rel'] = df_pla['t'] - t_start

    # 3. NỘI SUY (Bây giờ truyền vào mảng NumPy nên sẽ không bị lỗi nữa)
    interp_plan = {
        'x': np.interp(df_act['t_rel'], df_pla['t_rel'], df_pla['x']),
        'y': np.interp(df_act['t_rel'], df_pla['t_rel'], df_pla['y']),
        'z': np.interp(df_act['t_rel'], df_pla['t_rel'], df_pla['z']),
    }

    has_quat = all(col in df_act and col in df_pla for col in ['qx', 'qy', 'qz', 'qw'])
    has_euler = all(col in df_act and col in df_pla for col in ['roll_deg', 'pitch_deg', 'yaw_deg'])

    interp_euler = {}
    euler_act = {}
    if has_quat:
        euler_act = quaternions_to_euler(df_act)
        euler_pla = quaternions_to_euler(df_pla)
        for key in ['roll', 'pitch', 'yaw']:
            interp_euler[key] = np.interp(df_act['t_rel'], df_pla['t_rel'], euler_pla[key])
    elif has_euler:
        for col in ['roll_deg', 'pitch_deg', 'yaw_deg']:
            key = col.replace('_deg', '')
            interp_euler[key] = np.deg2rad(
                np.interp(df_act['t_rel'], df_pla['t_rel'], df_pla[col])
            )
            euler_act[key] = np.deg2rad(df_act[col])

    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(3, 3, wspace=0.35, hspace=0.4)

    def fix_axis(ax, title, ylabel):
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

    # --- ĐỒ THỊ THEO THỜI GIAN: VỊ TRÍ ---
    ax_x = fig.add_subplot(grid[0, 0])
    ax_x.plot(df_act['t_rel'], interp_plan['x'], 'r--', label='Reference')
    ax_x.plot(df_act['t_rel'], df_act['x'], 'b-', label='Actual', alpha=0.7)
    fix_axis(ax_x, "Tọa độ X", "X (m)")

    ax_y = fig.add_subplot(grid[1, 0])
    ax_y.plot(df_act['t_rel'], interp_plan['y'], 'r--', label='Reference')
    ax_y.plot(df_act['t_rel'], df_act['y'], 'g-', label='Actual', alpha=0.7)
    fix_axis(ax_y, "Tọa độ Y", "Y (m)")

    ax_z = fig.add_subplot(grid[2, 0])
    ax_z.plot(df_act['t_rel'], interp_plan['z'], 'r--', label='Reference')
    ax_z.plot(df_act['t_rel'], df_act['z'], 'm-', label='Actual', alpha=0.7)
    ax_z.set_xlabel("Thời gian (s)")
    fix_axis(ax_z, "Tọa độ Z", "Z (m)")

    # --- ĐỒ THỊ THEO THỜI GIAN: GÓC EULER ---
    if has_quat or has_euler:
        euler_specs = [
            ('roll', 'Góc Roll', 'Roll (rad)', 'tab:orange'),
            ('pitch', 'Góc Pitch', 'Pitch (rad)', 'tab:cyan'),
            ('yaw', 'Góc Yaw', 'Yaw (rad)', 'tab:brown'),
        ]
        for idx, (key, title, ylabel, color) in enumerate(euler_specs):
            ax = fig.add_subplot(grid[idx, 1])
            if key == 'yaw':
                ref_signal = np.zeros_like(df_act['t_rel'])
                act_signal = euler_act[key] - interp_euler[key]
            else:
                ref_signal = interp_euler[key]
                act_signal = euler_act[key]
            ax.plot(df_act['t_rel'], ref_signal, 'r--', label='Reference')
            ax.plot(df_act['t_rel'], act_signal, color=color, label='Actual', alpha=0.8)
            if idx == 2:
                ax.set_xlabel("Thời gian (s)")
            fix_axis(ax, title, ylabel)
    else:
        ax_note = fig.add_subplot(grid[:, 1])
        ax_note.axis('off')
        ax_note.text(
            0.5,
            0.5,
            "Không tìm thấy đủ cột quaternion hoặc Euler\n(`qx`,`qy`,`qz`,`qw` hoặc `roll_deg`,`pitch_deg`,`yaw_deg`)",
            ha='center',
            va='center',
            fontsize=12,
            color='dimgray'
        )

    # --- ĐỒ THỊ 3D ---
    ax_3d = fig.add_subplot(grid[:, 2], projection='3d')
    ax_3d.plot(df_pla['x'], df_pla['y'], df_pla['z'], 'r--', label='Planned Path', linewidth=2)
    ax_3d.plot(df_act['x'], df_act['y'], df_act['z'], 'b-', label='Actual Path', alpha=0.6)
    ax_3d.set_title("Quỹ đạo bay 3D", fontsize=14)
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.legend()

    plt.show()

except Exception as e:
    import traceback
    print(f"Lỗi cụ thể:\n{traceback.format_exc()}")