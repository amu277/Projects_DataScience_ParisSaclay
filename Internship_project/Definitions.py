from __future__ import division
import math
import pylab
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

### TRANSFORM RAW DATA

def transform_units(df, acc_ind, gyr_ind, mag_ind):
    if acc_ind == 1:  # Accelerometer calibrated

        acc_sens = 1
        # conversion of accelerometer data from mG to G
        df['acc_x'] = (df['acc_x'] * acc_sens) / 1000
        df['acc_y'] = (df['acc_y'] * acc_sens) / 1000
        df['acc_z'] = (df['acc_z'] * acc_sens) / 1000

    else:

        acc_sens = 0.122  # ACCELEROMETER_SENSITIVITY = 0.122 mg / LSB sensitivity
        # conversion of accelerometer data from mG to G
        df['acc_x'] = (df['acc_x'] * acc_sens) / 1000
        df['acc_y'] = (df['acc_y'] * acc_sens) / 1000
        df['acc_z'] = (df['acc_z'] * acc_sens) / 1000

    if gyr_ind == 1:  # Gyroscope calibrated

        gyr_sens = 1

        # conversion of gyroscope data
        df['gyr_x'] = (df['gyr_x'] * gyr_sens) / 1000
        df['gyr_y'] = (df['gyr_y'] * gyr_sens) / 1000
        df['gyr_z'] = (df['gyr_z'] * gyr_sens) / 1000

    else:

        gyr_sens = 70  # GYROSCOPE_SENSITIVITY = 70 mdeg / sec / LSB

        df['gyr_x'] = (df['gyr_x'] * gyr_sens) / 1000
        df['gyr_y'] = (df['gyr_y'] * gyr_sens) / 1000
        df['gyr_z'] = (df['gyr_z'] * gyr_sens) / 1000

    if mag_ind == 1:  # Magnetometer calibrated, sensitivity applied

        pass

    else:  # Convert magnetometer uncalibrated, sensitivity applied

        # Offsets
        off_x = 652.250732421875
        off_y = -39.21294021606445
        off_z = 17.11958122253418

        # Gains
        gain_yx = 0.051862746477127075
        gain_x = 1.4430811405181885
        gain_zy = -0.017628459259867668
        gain_xz = 0.07353414595127106
        gain_y = 1.6025787591934204
        gain_yz = -0.035460613667964935
        gain_xy = 0.04795261099934578
        gain_zx = 0.04007400572299957
        gain_z = 0.7521066665649414

        # New magnetometer values = (Mag - offsets) * Gains

        df['mag_x'] = (df['mag_x'] - off_x) * gain_x + (df['mag_y'] - off_y) * gain_xy + (df['mag_z'] - off_z) * gain_xz

        df['mag_y'] = (df['mag_x'] - off_x) * gain_yx + (df['mag_y'] - off_y) * gain_y + (df['mag_z'] - off_z) * gain_yz

        df['mag_z'] = (df['mag_x'] - off_x) * gain_zx + (df['mag_y'] - off_y) * gain_zy + (df['mag_z'] - off_z) * gain_z

    # Normalise accelerometer/magnetometer data -- check if this is done properly

    acc_n = df.acc_x * df.acc_x + df.acc_y * df.acc_y + df.acc_z * df.acc_z
    df['acc_norm'] = acc_n.apply(math.sqrt)

    df['acc_x_norm'] = df.acc_x / df.acc_norm
    df['acc_y_norm'] = df.acc_y / df.acc_norm
    df['acc_z_norm'] = df.acc_z / df.acc_norm

    mag_n = df.mag_x * df.mag_x + df.mag_y * df.mag_y + df.mag_z * df.mag_z
    df['mag_norm'] = mag_n.apply(math.sqrt)

    df['mag_x_norm'] = df.mag_x / df.mag_norm
    df['mag_y_norm'] = df.mag_y / df.mag_norm
    df['mag_z_norm'] = df.mag_z / df.mag_norm

    return df


### COMPUTE ROTATION ANGLES

def angles_comp(df, hertz):
    dt = 1/hertz # usually 20 ms sample rate, 50 HZ, or 1/50 = 0.02
    shape_y = df.shape[0]

    # set values
    accX = df['acc_x']
    accY = df['acc_y']
    accZ = df['acc_z']

    gyrX = df['gyr_x']
    gyrY = df['gyr_y']
    gyrZ = df['gyr_z']

    angles_df = pd.DataFrame()

    mu = 0.1

    roll = -gyrX.iloc[0] * dt
    pitch = -gyrY.iloc[0] * dt
    gyr_angle_z = -gyrZ.iloc[0] * dt

    roll_acc = (math.atan2(float(accY.iloc[0]),
                           float(math.sqrt(
                               mu * (accX.iloc[0] * accX.iloc[0]) + accZ.iloc[0] * accZ.iloc[0])))) * 180 / math.pi

    pitch_acc = (math.atan2(float(-1*accX.iloc[0]),
                            float(
                                math.sqrt(accY.iloc[0] * accY.iloc[0] + accZ.iloc[0] * accZ.iloc[0])))) * 180 / math.pi

    acc_z_tilt = math.atan2(float(accZ.iloc[0]), float(math.sqrt(
        accX.iloc[0] * accX.iloc[0] + accY.iloc[0] * accY.iloc[0] + accZ.iloc[0] * accZ.iloc[0]))) * 180 / math.pi

    angles = pd.Series([roll, pitch, gyr_angle_z, roll_acc, pitch_acc, acc_z_tilt, gyrX.iloc[0], gyrY.iloc[0], gyrZ.iloc[0], accX.iloc[0], accY.iloc[0], accZ.iloc[0]],
                       ['gyr_roll_x', 'gyr_pitch_y', 'gyr_angle_z', 'acc_roll_x', 'acc_pitch_y',
                        'acc_angle_z', 'gyr_x', 'gyr_y', 'gyr_z', 'acc_x', 'acc_y', 'acc_z'])

    angles_df = angles_df.append([angles], ignore_index=True)

    for i, j in zip(range(1, shape_y), range(1, shape_y)):

        # Integrate the gyroscope data -> int(angularSpeed) = angle, quick and dirty way

        roll += -gyrX.iloc[j] * dt;
        pitch += -gyrY.iloc[j] * dt;
        gyr_angle_z += -gyrZ.iloc[j] * dt;

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        # Accelerometer angles calculation

        # X axis rotation - roll
        roll_acc = (math.atan2(float(accY.iloc[i]),
                               float(math.sqrt(
                                   mu * (accX.iloc[i] * accX.iloc[i]) + accZ.iloc[i] * accZ.iloc[i])))) * 180 / math.pi

        # Y axis rotation - pitch
        pitch_acc = (math.atan2(float(-1*accX.iloc[i]), float(math.sqrt(
            accY.iloc[i] * accY.iloc[i] + accZ.iloc[i] * accZ.iloc[
                i])))) * 180 / math.pi  # arctan(-Gpx / sqrt(Gpy^2 + Gpz^2))

        # Tilt angle Z - if no linear acc
        acc_z_tilt = math.atan2(float(accZ.iloc[i]), float(math.sqrt(
            accX.iloc[i] * accX.iloc[i] + accY.iloc[i] * accY.iloc[i] + accZ.iloc[i] * accZ.iloc[i]))) * 180 / math.pi
        # arctan(Gpz/ sqrt(Gpx^2 + Gpy^2 + Gpz^2))


        angles = pd.Series([roll, pitch, gyr_angle_z, roll_acc, pitch_acc, acc_z_tilt, gyrX.iloc[i], gyrY.iloc[i], gyrZ.iloc[i], accX.iloc[i], accY.iloc[i], accZ.iloc[i]],
                           ['gyr_roll_x', 'gyr_pitch_y', 'gyr_angle_z', 'acc_roll_x', 'acc_pitch_y',
                            'acc_angle_z', 'gyr_x', 'gyr_y', 'gyr_z', 'acc_x', 'acc_y', 'acc_z'])

        angles_df = angles_df.append([angles], ignore_index=True)

    return (angles_df)

# Kalman filter with fixed Q,R (for adaptive, remove q,r matrices setting in filter step)

def kalman_filter(df, vib_ind, hertz):
    dt = 1/hertz

    meas_list_x = []
    for i in range(0, df.shape[0]):
        n = [df.acc_roll_x[i], df.gyr_x[i]]
        meas_list_x.append(n)

    meas_list_y = []
    for i in range(0, df.shape[0]):
        n = [df.acc_pitch_y[i], df.gyr_y[i]]
        meas_list_y.append(n)

    # State transition matrix
    A = [[1, -dt],
         [0, 1]]

    # Observation matrix
    H = [[1, 0.0], [0.0, 1]]

    # Covariance matrices setting
    if vib_ind == 1:
        q = 0.001
        r = 0.05
    else:
        q = 0.01
        r = 8.

    # Transition covariance matrix - process
    Q = np.matrix([[q, 0.0], [0.0, q]])

    # Observation covariance matrix - meas
    # R = np.matrix([[saccY**2, 0.0], [0.0, sgyrY**2]])
    R = np.matrix([[r, 0.0], [0.0, r]])

    # Kalman filter on x-axis
    kfx = KalmanFilter(transition_matrices=A, observation_matrices=H, transition_covariance=Q, observation_covariance=R)
    measurements_x = np.array(meas_list_x)  # 2 observations
    kfx = kfx.em(measurements_x, n_iter=5)
    (filtered_state_means_x, filtered_state_covariances_x) = kfx.filter(measurements_x)
    # (smoothed_state_means, smoothed_state_covariances) = kfx.smooth(measurements)

    # Kalman filter on y-axis
    kfy = KalmanFilter(transition_matrices=A, observation_matrices=H, transition_covariance=Q, observation_covariance=R)
    measurements_y = np.array(meas_list_y)  # 2 observations
    kfy = kfy.em(measurements_y, n_iter=5)
    (filtered_state_means_y, filtered_state_covariances_y) = kfy.filter(measurements_y)
    # (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

    data_plot_x = []
    gyr_angle = meas_list_x[0][0]
    for i in range(len(meas_list_x)):
        # gyr_angle += -dt * meas_list_x[i][1]
        n = [meas_list_x[i][0], df.gyr_roll_x[i], filtered_state_means_x[i, 0]]
        data_plot_x.append(n)

    val_dfx = pd.DataFrame(data_plot_x)
    val_dfx.columns = ['acc_angle_x', 'gyr_roll_x', 'kalman_roll']
    val_dfx['ts'] = val_dfx.index

    data_plot_y = []
    gyr_angle = meas_list_y[0][0]
    for i in range(len(meas_list_y)):
        # gyr_angle += -dt * meas_list_y[i][1]
        n = [meas_list_y[i][0], df.gyr_pitch_y[i], filtered_state_means_y[i, 0]]
        data_plot_y.append(n)

    val_dfy = pd.DataFrame(data_plot_y)
    val_dfy.columns = ['acc_angle_y', 'gyr_pitch_y', 'kalman_pitch']
    val_dfy['ts'] = val_dfy.index

    val_df = pd.merge(val_dfx, val_dfy, how='inner', on='ts')

    return val_df


### YAW CALCULATION

def mag_yaw(kdf, mdf):
    # Equations to transform magnetometer to horizontal plane
    # Y = pitch angle in radians, X = roll angle in radians
    # Xh = mag_x*cos(Y)+mag_y*sin(Y)sin(X)+mag_z*sin(Y)*cos(X)
    # -Yh = mag_z*sin(X)-mag_y*cos(X)
    # yaw = atan2(-Yh/Xh) to get results between - pi and pi

    shape_y = kdf.shape[0]

    # mdf = mdf[:-1] # to match kalman filter

    mag_x = mdf['mag_x_norm']
    mag_y = mdf['mag_y_norm']
    mag_z = mdf['mag_z_norm']

    roll = kdf['kalman_roll'] * math.pi / 180  # roll radians
    pitch = kdf['kalman_pitch'] * math.pi / 180  # pitch radians

    angles_df = pd.DataFrame()

    # -Yh
    Yh = mag_z.iloc[0] * math.sin(roll.iloc[0]) - mag_y.iloc[0] * math.cos(roll.iloc[0])

    # Xh
    Xh = mag_x.iloc[0] * math.cos(pitch.iloc[0]) + mag_y.iloc[0] * math.sin(pitch.iloc[0]) * math.sin(roll.iloc[0]) + \
         mag_z.iloc[0] * math.sin(pitch.iloc[0]) * math.cos(roll.iloc[0])

    yaw = -(math.atan2(float(Yh), float(Xh))) * (180 / math.pi)

    yaw_simple = (math.atan2(float(mag_y.iloc[0]), float(mag_x.iloc[0]))) * (180 / math.pi)

    angles0 = pd.Series([kdf.acc_angle_x.iloc[0], kdf.gyr_roll_x.iloc[0], kdf.kalman_roll.iloc[0],
                         kdf.acc_angle_y.iloc[0], kdf.gyr_pitch_y.iloc[0], kdf.kalman_pitch.iloc[0],
                         yaw, Xh, Yh, mdf.mag_x_norm.iloc[0], mdf.mag_y_norm.iloc[0], mdf.mag_z_norm.iloc[0],
                         yaw_simple],
                        ['acc_angle_x', 'gyr_roll_x', 'kalman_roll',
                         'acc_angle_y', 'gyr_pitch_y', 'kalman_pitch',
                         'yaw', 'Xh', 'Yh', 'mag_x_norm', 'mag_y_norm', 'mag_z_norm', 'yaw_simple'])

    angles_df = angles_df.append([angles0], ignore_index=True)

    for i in range(1, shape_y):
        # Calculate headings on each axis

        # -Yh
        Yh += mag_z.iloc[i] * math.sin(roll.iloc[i]) - mag_y.iloc[i] * math.cos(roll.iloc[i])

        # Xh
        Xh += mag_x.iloc[i] * math.cos(pitch.iloc[i]) + mag_y.iloc[i] * math.sin(pitch.iloc[i]) * math.sin(
            roll.iloc[i]) + mag_z.iloc[i] * math.sin(pitch.iloc[i]) * math.cos(roll.iloc[i])

        # Compute yaw

        yaw = -(math.atan2(float(Yh), float(Xh))) * (180 / math.pi)  # add magnetic declination for paris, +1.61 ?

        yaw_simple = (math.atan2(float(mag_y.iloc[i]), float(mag_x.iloc[i]))) * (180 / math.pi)

        # print(Xh, Yh, yaw)

        angles = pd.Series([kdf.acc_angle_x.iloc[i], kdf.gyr_roll_x.iloc[i], kdf.kalman_roll.iloc[i],
                            kdf.acc_angle_y.iloc[i], kdf.gyr_pitch_y.iloc[i], kdf.kalman_pitch.iloc[i],
                            yaw, Xh, Yh, mdf.mag_x_norm.iloc[i], mdf.mag_y_norm.iloc[i], mdf.mag_z_norm.iloc[i],
                            yaw_simple],
                           ['acc_angle_x', 'gyr_roll_x', 'kalman_roll',
                            'acc_angle_y', 'gyr_pitch_y', 'kalman_pitch',
                            'yaw', 'Xh', 'Yh', 'mag_x_norm', 'mag_y_norm', 'mag_z_norm', 'yaw_simple'])

        angles_df = angles_df.append([angles], ignore_index=True)

    return angles_df

### EULER ANGLES TO QUATERNION CONVERSION

def eul2quat(df):
    # orientation xyz
    cos = math.cos
    sin = math.sin

    quat_df = pd.DataFrame()

    len_df = df.shape[0] - 1

    for i in range(0, len_df):
        r = float(df.x_rot[i]) / 2;
        p = float(df.y_rot[i]) / 2;
        y = float(df.z_rot[i]) / 2;

        q = []

        w = (cos(y) * cos(p) * cos(r)) + (sin(y) * sin(p) * sin(r));  # w - orientation
        x = (cos(y) * cos(p) * sin(r)) - (sin(y) * sin(p) * cos(r));  # x
        y = (cos(y) * sin(p) * cos(r)) + (sin(y) * cos(p) * sin(r));  # y
        z = (sin(y) * cos(p) * cos(r)) - (cos(y) * sin(p) * sin(r));  # z

        quat = pd.Series([w, x, y, z], ['w', 'x', 'y', 'z'])

        quat_df = quat_df.append([quat], ignore_index=True)

    return quat_df


### ZONE TO QUADRANT LABEL

def zone_df():
    # Zones list
    zdf = [{'zone_name': 'Up_mol_le_ext', 'quad_name': 'Up_left', 'quad_nbr': 0},  # Upper left molar
           {'zone_name': 'Lo_mol_le_ext', 'quad_name': 'Lo_left', 'quad_nbr': 1},  # Lower left molar
           {'zone_name': 'Up_mol_ri_ext', 'quad_name': 'Up_right', 'quad_nbr': 2},  # Upper right molar
           {'zone_name': 'Lo_mol_ri_ext', 'quad_name': 'Lo_right', 'quad_nbr': 3},  # Lower right molar
           {'zone_name': 'Up_inc_ext', 'quad_name': 'Inc_ext', 'quad_nbr': 4},
           {'zone_name': 'Lo_inc_ext', 'quad_name': 'Inc_ext', 'quad_nbr': 4},
           {'zone_name': 'Up_mol_le_occ', 'quad_name': 'Up_left', 'quad_nbr': 0},  # Upper left molar
           {'zone_name': 'Up_mol_le_int', 'quad_name': 'Up_left', 'quad_nbr': 0},  # Upper left molar
           {'zone_name': 'Lo_mol_le_int', 'quad_name': 'Lo_left', 'quad_nbr': 1},  # Lower left molar
           {'zone_name': 'Lo_mol_le_occ', 'quad_name': 'Lo_left', 'quad_nbr': 1},  # Lower left molar
           {'zone_name': 'Up_mol_ri_occ', 'quad_name': 'Up_right', 'quad_nbr': 2},  # Upper right molar
           {'zone_name': 'Up_mol_ri_int', 'quad_name': 'Up_right', 'quad_nbr': 2},  # Upper right molar
           {'zone_name': 'Lo_mol_ri_int', 'quad_name': 'Lo_right', 'quad_nbr': 3},  # Lower right molar
           {'zone_name': 'Lo_mol_ri_occ', 'quad_name': 'Lo_right', 'quad_nbr': 3},  # Lower right molar
           {'zone_name': 'Up_inc_int', 'quad_name': 'Inc_int', 'quad_nbr': 5},
           {'zone_name': 'Lo_inc_int', 'quad_name': 'Inc_int', 'quad_nbr': 5},
           {'zone_name': 'unknown', 'quad_name': 'unknown', 'quad_nbr': 6}
           ]

    zdf = pd.DataFrame(zdf)
    zdf['zone_nbr'] = zdf.index

    return zdf

### REMOVE LOCATION OUTLIERS ITERATIVELY

# array = np.array(df[['kalman_roll', 'kalman_pitch', 'yaw', 'ts']])

def rem_outliers(array, stdev): # Input array, standard deviation at which to remove (suggested = 4)

    # Iteratively remove outliers, starting with z, then x, then y

    # Remove z outliers
    m = stdev
    d = np.abs(array[:,2] - np.median(array[:,2])) # find absolute distance for yaw values from median
    mdev = np.median(d) # find median distance
    s = d/(mdev if mdev else 1.) # standard deviation calculation
    zarr = array[s<m]

    # Remove x outliers
    m = stdev
    d = np.abs(zarr[:,0] - np.median(zarr[:,0])) # find absolute distance for yaw values from median
    mdev = np.median(d) # find median distance
    s = d/(mdev if mdev else 1.) # standard deviation calculation
    xarr =  array[s<m]

    # Remove y outliers
    m = stdev
    d = np.abs(xarr[:,1] - np.median(xarr[:,1])) # find absolute distance for yaw values from median
    mdev = np.median(d) # find median distance
    s = d/(mdev if mdev else 1.) # standard deviation calculation
    farr =  array[s<m]

    return farr
