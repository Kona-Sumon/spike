import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.linalg import block_diag

file_path = 'indy_20161005_06.mat'

# 任务1：绘制raster图
def plot_raster_batch_adjusted(all_units, channel_range, batch_number, start_time, title='Raster Plot Batch'):
    # 计算需要绘制的图的数量
    num_plots = sum(len(all_units[ch]) for ch in channel_range)
    
    # 创建子图
    fig, axs = plt.subplots(num_plots, 1, figsize=(20, 0.5 * num_plots), sharex=True)

    unit_count = 0
    # 遍历指定通道范围
    for channel_index in channel_range:
        channel_units = all_units[channel_index]
        # 遍历通道中的单元
        for unit_index, spikes in enumerate(channel_units):
            # 仅保留大于等于指定起始时间的脉冲时间
            spikes = spikes[spikes >= start_time]
            y = [unit_count] * len(spikes)
            
            # 绘制scatter图
            axs[unit_count].scatter(spikes, y, marker='|', color='black')
            axs[unit_count].set_yticks([])
            
            # 隐藏y轴刻度和图框
            axs[unit_count].grid(False)
            for spine in axs[unit_count].spines.values():
                spine.set_visible(False)
            
            # 更新单位计数
            unit_count += 1

    # 设置图的标题和X轴ticks
    fig.suptitle(f'{title} - All Channels', fontsize=16)
    axs[0].set_xticks([])
    
    # 调整子图之间的垂直间距
    plt.subplots_adjust(hspace=0)
    
    # 显示图形
    plt.show()


def extract_all_sorted_units(file, spikes_data):
    # 存储所有排序单元的脉冲时间
    all_sorted_unit_spike_times = []
    
    # 遍历所有通道
    for channel_index in range(spikes_data.shape[1]):
        # 存储当前通道的排序单元脉冲时间
        channel_sorted_units = []
        
        # 遍历所有单元
        for unit_index in range(1, spikes_data.shape[0]):
            # 获取单元对应的引用
            ref = spikes_data[unit_index, channel_index]
            
            # 检查引用是否存在
            if ref:
                # 从文件中提取脉冲数据并展平为一维数组
                spikes = file[ref][:].flatten()
                
                # 检查脉冲数据是否非空
                if spikes.size > 0:
                    # 存储排序单元的脉冲时间
                    channel_sorted_units.append(spikes)
        
        # 将当前通道的排序单元脉冲时间添加到总列表中
        all_sorted_unit_spike_times.append(channel_sorted_units)
    
    # 返回所有通道的排序单元脉冲时间列表
    return all_sorted_unit_spike_times

# 读取HDF5文件
with h5py.File(file_path, 'r') as file:
    # 读取数据
    chan_names = file['chan_names'][:]
    cursor_pos = file['cursor_pos'][:]
    finger_pos = file['finger_pos'][:]
    spikes = file['spikes'][:]
    t = file['t'][:]
    target_pos = file['target_pos'][:]
    wf = file['wf'][:]

    # 提取所有排序后的单元的脉冲时间
    all_sorted_spike_times = extract_all_sorted_units(file, file['spikes'])
    all_units_spikes = []

    # 遍历所有单位的脉冲数据
    for i in range(spikes.shape[0]):
        unit_spikes = []
        for j in range(spikes.shape[1]):
            ref = spikes[i, j]
            if ref:
                spikes_data = file[ref][:]
                unit_spikes.append(spikes_data)
        all_units_spikes.append(unit_spikes)


# 计算需要分批处理的通道数量等信息
num_channels = len(all_sorted_spike_times)
num_sorted_units = sum(len(channel_units) for channel_units in all_sorted_spike_times)
batch_size = 96
num_batches = (num_channels // batch_size) + (0 if num_channels % batch_size == 0 else 1)  # 需要的总批次数
batch_number = 1
channel_range = range((batch_number - 1) * batch_size, min(batch_number * batch_size, num_channels))
start_time = 1250

# 绘制批次对应的raster图
plot_raster_batch_adjusted(all_sorted_spike_times, channel_range, batch_number, start_time)

##################################################################################

# 任务1：绘制tuning curve图 (位置)
# 提取目标位置的x和y坐标
target_x_positions = target_pos[0, :]
target_y_positions = target_pos[1, :]

# 计算角度并转换为0到360度的范围
angles = np.degrees(np.arctan2(target_y_positions, target_x_positions))
angles = np.mod(angles, 360)

# 设置方向的数量和方向的大小
num_direction_bins = 8
bin_size = 360 / num_direction_bins

# 创建方向的bins
direction_bins = np.arange(0, 360, bin_size)

# 计算直方图
hist, edges = np.histogram(angles, bins=direction_bins)
bin_centers = (edges[:-1] + edges[1:]) / 2

# 使用UnivariateSpline进行平滑拟合
spline = UnivariateSpline(bin_centers, hist, s=0.5)  # s是平滑因子
x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
y_smooth = spline(x_smooth)

# 绘制平滑拟合后的图形
plt.plot(x_smooth, y_smooth, label='Tuning Curve')
plt.title('Tuning Curve of Movement Directions')
plt.xlabel('Direction (degrees)')
plt.ylabel('Frequency')

# 设置X轴刻度为方向的bins
plt.xticks(direction_bins)

# 添加图例
plt.legend()

# 显示图形
plt.show()

############################################################

# 任务1：绘制tuning curve图 (速度)
# 将时间展平
t = np.ravel(t)

# 计算光标位置的变化
delta_pos = np.diff(cursor_pos, axis=1)
delta_t = np.diff(t)
delta_t[delta_t == 0] = np.finfo(float).eps

# 计算光标速度
velocity = np.sqrt((delta_pos[0] / delta_t)**2 + (delta_pos[1] / delta_t)**2)

# 将手指位置从厘米转换为毫米
finger_pos_mm = finger_pos * 10
finger_pos_xy = finger_pos_mm[:2, :]

# 计算手指速度，即位置的时间导数
delta_finger_pos = np.diff(finger_pos_xy, axis=1)
finger_velocity = np.sqrt((delta_finger_pos[0] / delta_t)**2 +
                          (delta_finger_pos[1] / delta_t)**2)

# 绘制手指速度的直方图
plt.hist(finger_velocity, bins=50, alpha=0.75)
plt.title('Finger Velocity Distribution')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Frequency')
plt.show()

# 提取排序后的脉冲时间
sorted_units_spikes = all_units_spikes[1:]
total_time = t[-1] - t[0]

# 计算每个单元的放电率
discharge_rates = []
for unit in sorted_units_spikes:
    unit_rates = []
    for channel_spikes in unit:
        rate = len(channel_spikes) / total_time
        unit_rates.append(rate)
    discharge_rates.append(unit_rates)

# 设置速度直方图的分箱数量
num_bins = 20
velocity_bins, bin_edges = np.histogram(finger_velocity, bins=num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 将排序后的脉冲时间展平
sorted_spike_times_flat = np.concatenate([
    spikes.flatten() for unit in sorted_units_spikes[1:] for channel in unit
    for spikes in channel if spikes.size > 0
])

# 查找脉冲时间对应的索引
sorted_spike_indices = np.searchsorted(t[1:], sorted_spike_times_flat)
sorted_spike_indices = sorted_spike_indices[sorted_spike_indices < len(finger_velocity)]

# 将脉冲时间分配到速度直方图的分箱中
sorted_spike_bins = np.digitize(finger_velocity[sorted_spike_indices], bin_edges) - 1
sorted_spikes_per_bin = np.zeros(num_bins)
sorted_time_per_bin = np.zeros(num_bins)

# 统计每个分箱中的脉冲数量和时间
for idx, bin_idx in enumerate(sorted_spike_bins):
    if 0 <= bin_idx < num_bins:
        sorted_spikes_per_bin[bin_idx] += 1
        sorted_time_per_bin[bin_idx] += delta_t[sorted_spike_indices[idx] - 1]

# 避免除零错误
sorted_time_per_bin[sorted_time_per_bin == 0] = np.finfo(float).eps

# 计算每个分箱的平均放电率
sorted_average_rates_per_bin = sorted_spikes_per_bin / sorted_time_per_bin

# 绘制神经元调谐曲线基于手指速度（仅限排序单元）
plt.plot(bin_centers, sorted_average_rates_per_bin, marker='o')
plt.title('Neuronal Tuning Curve Based on Finger Velocity (Sorted Units Only)')
plt.xlabel('Finger Velocity (mm/s)')
plt.ylabel('Average Discharge Rate (spikes/s)')
plt.show()

##################################################################

# 任务2：对速度编码模型的拟合程度
# 将速度值和放电率准备为模型输入
y = sorted_average_rates_per_bin  # 放电率
X = bin_centers.reshape(-1, 1)  # 速度值

# 建立线性回归模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)  # 计算预测值

# 计算R^2分数
r_squared = r2_score(y, y_pred)
print("r_squared =", r_squared)  # 输出模型的R^2

# 绘制实际放电率和模型预测放电率的关系
plt.scatter(X, y, color='black', label='Actual Discharge Rates')
plt.plot(X, y_pred, color='blue', linewidth=2, label='Predicted Discharge Rates')
plt.title('Neuronal Discharge Rates vs. Velocity')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Discharge Rate (spikes/s)')
plt.legend()
plt.show()

##################################################################################

# 任务1：绘制PSTH直方图
# 定义前刺激和后刺激的时间以及直方图的宽度
pre_stimulus_time = 1.0
post_stimulus_time = 1.0
bin_width = 0.01

# 创建时间的bins
bins = np.arange(-pre_stimulus_time, post_stimulus_time, bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2

# 调整脉冲时间为相对于刺激的时间
adjusted_spike_times = sorted_spike_times_flat - t[0]

# 计算PSTH的直方图
psth_counts, _ = np.histogram(adjusted_spike_times, bins=bins)

# 归一化直方图频率
psth_rates = psth_counts / (bin_width * len(sorted_units_spikes[1:]))

# 绘制PSTH图
plt.bar(bin_centers, psth_rates, width=bin_width, color='grey')
plt.title('PSTH of Neuronal Discharge Rates')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

########################################################################################

# 任务1：绘制tuning curve图 (加速度)
# 计算手指速度的加速度
acceleration = np.diff(finger_velocity) / np.mean(delta_t)

# 调整脉冲时间的索引，确保不越界
adjusted_spike_indices = sorted_spike_indices[sorted_spike_indices < len(acceleration) + 1]

# 设置加速度的bins
acceleration_bins = np.linspace(np.min(acceleration), np.max(acceleration), num_bins + 1)
acceleration_bin_centers = (acceleration_bins[:-1] + acceleration_bins[1:]) / 2

# 初始化存储每个加速度bin的脉冲数量和时间的数组
spikes_per_acceleration_bin = np.zeros(num_bins)
time_in_acceleration_bins = np.zeros(num_bins)

# 遍历每个脉冲时间，将其分配到相应的加速度bin中
for spike_index in adjusted_spike_indices:
    acceleration_value = acceleration[spike_index - 1]
    bin_index = np.digitize(acceleration_value, acceleration_bins) - 1
    if 0 <= bin_index < num_bins:
        spikes_per_acceleration_bin[bin_index] += 1
        time_in_acceleration_bins[bin_index] += delta_t[spike_index - 1]

# 避免除零错误
time_in_acceleration_bins[time_in_acceleration_bins == 0] = np.finfo(float).eps

# 计算每个加速度bin的平均放电率
average_rates_per_acceleration_bin = spikes_per_acceleration_bin / time_in_acceleration_bins

# 绘制基于加速度的神经元调谐曲线
plt.plot(acceleration_bin_centers, average_rates_per_acceleration_bin, marker='o')
plt.title('Neuronal Tuning Curve Based on Acceleration')
plt.xlabel('Acceleration (mm/s²)')
plt.ylabel('Average Discharge Rate (spikes/s)')
plt.show()
