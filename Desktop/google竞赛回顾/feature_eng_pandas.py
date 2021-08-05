import pandas as pd
import numpy as np
import simdkalman

from pathlib import Path
from tqdm import tqdm

pd.set_option('expand_frame_repr', False)


def get_union(dir_name, file_name):
    print("Reading", file_name, "from", dir_name, "...")
    datas = []
    for path in tqdm(Path(dir_name).rglob(file_name)):
        datas.append(pd.read_csv(path, low_memory=False, index_col=False))
    return pd.concat(datas)


def parse_gnsslog():
    header_of_table_named = {
        "UncalMag":
            "utcTimeMillis,elapsedRealtimeNanos,UncalMagXMicroT,UncalMagYMicroT,UncalMagZMicroT\n",
        "UncalAccel":
            "utcTimeMillis,elapsedRealtimeNanos,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2\n",
        "UncalGyro":
            "utcTimeMillis,elapsedRealtimeNanos,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec\n",
        "Status":
            "UnixTimeMillis,SignalCount,SignalIndex,ConstellationType,Svid,CarrierFrequencyHz,Cn0DbHz,"
            "AzimuthDegrees,ElevationDegrees,UsedInFix,HasAlmanacData,HasEphemerisData\n",
        "Raw":
            "utcTimeMillis,TimeNanos,LeapSecond,TimeUncertaintyNanos,FullBiasNanos,BiasNanos,BiasUncertaintyNanos,"
            "DriftNanosPerSecond,DriftUncertaintyNanosPerSecond,HardwareClockDiscontinuityCount,"
            "Svid,TimeOffsetNanos,State,ReceivedSvTimeNanos,ReceivedSvTimeUncertaintyNanos,Cn0DbHz,"
            "PseudorangeRateMetersPerSecond,PseudorangeRateUncertaintyMetersPerSecond,AccumulatedDeltaRangeState,"
            "AccumulatedDeltaRangeMeters,AccumulatedDeltaRangeUncertaintyMeters,CarrierFrequencyHz,CarrierCycles,"
            "CarrierPhase,CarrierPhaseUncertainty,MultipathIndicator,SnrInDb,ConstellationType,AgcDb\n",
        "Fix":
            "Provider,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SpeedMps,AccuracyMeters,BearingDegrees,"
            "UnixTimeMillis,SpeedAccuracyMps,BearingAccuracyDegrees\n"
    }
    for part in ["./train", "./test"]:
        print(part, "gnsslog parsing ...")
        for file_name in Path(part).rglob("*GnssLog.txt"):
            print(file_name)
            with open(str(file_name)) as f_open:
                datalines = f_open.readlines()
            f_named = {}
            for f_name in header_of_table_named.keys():
                if not (file_name.parent / (f_name + ".csv")).exists():
                    f_named[f_name] = open(file_name.parent / (f_name + ".csv"), "w")
            if f_named:
                collectionName = file_name.parent.parent.name
                phoneName = file_name.parent.name
                for f_name, f in f_named.items():
                    f.write("collectionName,phoneName," + header_of_table_named[f_name])
                for dataline in tqdm(datalines):
                    for f_name in f_named.keys():
                        if dataline.startswith(f_name):
                            f_named[f_name].write(collectionName)
                            f_named[f_name].write(',')
                            f_named[f_name].write(phoneName)
                            f_named[f_name].write(',')
                            f_named[f_name].write(dataline[len(f_name) + 1:])
                            break
                for f in f_named.values():
                    f.close()
    print("finish")


def calc_haversine(lat1, lng1, lat2, lng2):
    """from
    https://www.kaggle.com/dehokanta/baseline-post-processing-by-outlier-correction
    """
    RADIUS = 6_367_000
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    dist = 2 * RADIUS * np.arcsin(a ** 0.5)
    return dist


class KF:
    def __init__(self):
        T = 1.0
        state_transition = np.array([[1, 0, T, 0, 0.5 * T ** 2, 0],
                                     [0, 1, 0, T, 0, 0.5 * T ** 2],
                                     [0, 0, 1, 0, T, 0],
                                     [0, 0, 0, 1, 0, T],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])
        process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]) + np.ones((6, 6)) * 1e-9
        observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9
        self.kf = simdkalman.KalmanFilter(
            state_transition=state_transition,
            process_noise=process_noise,
            observation_model=observation_model,
            observation_noise=observation_noise)

    def apply_kf_smoothing(self, df, suffix):
        phones = df["phone"].drop_duplicates().tolist()
        for phone in tqdm(phones):
            cond = df['phone'] == phone
            tmp = df[cond].copy()
            tmp[0] = tmp["millisSinceGpsEpoch"] // 1000
            tmp = tmp.merge(pd.DataFrame(range(tmp[0].min(), tmp[0].max() + 1)), on=[0], how="right")
            tmp_np = tmp[['latDeg', 'lngDeg']].to_numpy()
            nan_idxs = tmp[tmp["millisSinceGpsEpoch"].isnull()].index.to_list()
            tmp_np = tmp_np.reshape(1, len(tmp_np), 2)
            smoothed = self.kf.smooth(tmp_np).states.mean
            smoothed = np.delete(smoothed, list(nan_idxs), 1)
            df.loc[cond, 'latDeg' + suffix] = smoothed[0, :, 0]
            df.loc[cond, 'lngDeg' + suffix] = smoothed[0, :, 1]


if __name__ == "__main__":
    DATA_DIR = Path("./")

    # 读取
    parse_gnsslog()
    train_raw_data = get_union(DATA_DIR / "train", "Raw.csv")
    test_raw_data = get_union(DATA_DIR / "test", "Raw.csv")
    train_gyro_data = get_union(DATA_DIR / "train", "UncalGyro.csv")
    test_gyro_data = get_union(DATA_DIR / "test", "UncalGyro.csv")
    train_mag_data = get_union(DATA_DIR / "train", "UncalMag.csv")
    test_mag_data = get_union(DATA_DIR / "test", "UncalMag.csv")
    train_accel_data = get_union(DATA_DIR / "train", "UncalAccel.csv")
    test_accel_data = get_union(DATA_DIR / "test", "UncalAccel.csv")

    gt_data = get_union(DATA_DIR / "train", "ground_truth.csv")
    train_data = pd.read_csv(DATA_DIR / "baseline_locations_train.csv")
    test_data = pd.read_csv(DATA_DIR / "baseline_locations_test.csv")
    sub_data = pd.read_csv(DATA_DIR / "sample_submission.csv")

    train_derived_data = get_union(DATA_DIR / "train", "*_derived.csv")
    test_derived_data = get_union(DATA_DIR / "test", "*_derived.csv")

    # gt合并
    gt_data.rename(columns={
        "latDeg": "latDeg_truth",
        "lngDeg": "lngDeg_truth",
        "heightAboveWgs84EllipsoidM": "heightAboveWgs84EllipsoidM_truth"}, inplace=True)
    train_data = train_data.merge(gt_data, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'])

    # 计算baseline和truth的距离
    train_data["dist_between_baseline_and_truth"] = calc_haversine(
        train_data.latDeg, train_data.lngDeg,
        train_data.latDeg_truth, train_data.lngDeg_truth
    )

    # 手机类别
    idx_of_phonename = {phoneName: idx for idx, phoneName in enumerate(test_data.phoneName.drop_duplicates().to_list())}
    train_data["phoneCat"] = train_data["phoneName"].map(idx_of_phonename)
    test_data["phoneCat"] = test_data["phoneName"].map(idx_of_phonename)


    def data_processing(data):
        # 时间周期性特征
        data[["month", "day"]] = data['collectionName'].str.split('-', expand=True)[[1, 2]]
        data["ratio_of_year"] = (30 * (data.month.astype(int) - 1) + data.day.astype(int)) / 365

        # 相对于相邻帧平均值的差值的绝对值
        offsets = [1, 2, 3, 4, 5]
        groups = data[["phone", "latDeg", "lngDeg"]].groupby("phone")
        for offset in offsets:
            data[["mean_latDeg+-" + str(offset), "mean_lngDeg+-" + str(offset)]] = \
                groups.rolling(window=offset * 2 + 1, min_periods=2, center=True).mean().values
            data["latDeg_mean_delta_" + str(offset)] = np.abs(data["mean_latDeg+-" + str(offset)] - data["latDeg"])
            data["lngDeg_mean_delta_" + str(offset)] = np.abs(data["mean_lngDeg+-" + str(offset)] - data["lngDeg"])

        # 添加前后差值特征
        data[["latDeg-1", "lngDeg-1"]] = groups.shift(1)
        data[["latDeg+1", "lngDeg+1"]] = groups.shift(-1)
        data["latDeg_pre_increment"] = data["latDeg"] - data["latDeg-1"]
        data["lngDeg_pre_increment"] = data["lngDeg"] - data["lngDeg-1"]
        data["latDeg_post_increment"] = data["latDeg+1"] - data["latDeg"]
        data["lngDeg_post_increment"] = data["lngDeg+1"] - data["lngDeg"]
        data["dist_pre"] = calc_haversine(data["latDeg"], data["lngDeg"], data["latDeg-1"], data["lngDeg-1"])
        data["dist_post"] = calc_haversine(data["latDeg"], data["lngDeg"], data["latDeg+1"], data["lngDeg+1"])

        # 没有前一帧，用后一帧来代替
        nan_idxs = data[data.latDeg_pre_increment.isnull()].index
        data.loc[nan_idxs, ["latDeg_pre_increment", "lngDeg_pre_increment", "dist_pre"]] = \
            data.loc[nan_idxs + 1, ["latDeg_pre_increment", "lngDeg_pre_increment", "dist_pre"]].values
        nan_idxs = data[data.latDeg_post_increment.isnull()].index
        data.loc[nan_idxs, ["latDeg_post_increment", "lngDeg_post_increment", "dist_post"]] = \
            data.loc[nan_idxs - 1, ["latDeg_post_increment", "lngDeg_post_increment", "dist_post"]].values

        # 添加前后差值绝对均值特征
        data["latDeg_pre_post_mean_abs_delta"] = (np.abs(data["latDeg_pre_increment"]) +
                                                  np.abs(data["latDeg_post_increment"])) / 2
        data["lngDeg_pre_post_mean_abs_delta"] = (np.abs(data["lngDeg_pre_increment"]) +
                                                  np.abs(data["lngDeg_post_increment"])) / 2
        data["pre_post_mean_abs_dist"] = (np.abs(data["dist_pre"]) + np.abs(data["dist_post"])) / 2

        # 同一时间不同设备的统计数值
        col_names = ["latDeg", "lngDeg", "heightAboveWgs84EllipsoidM"]
        fn_names = ["max", "min", "mean", "sum", "count"]
        groups = data.groupby("millisSinceGpsEpoch")
        for fn_name_ in fn_names:
            data[[fn_name_ + '_' + col_name + "_with_same_millisSinceGpsEpoch" for col_name in col_names]] = \
                groups[col_names].transform(fn_name_).values

        # 与同时间所有手机位置均值的距离
        data["dist_between_baseline_and_phone_mean"] = calc_haversine(data.latDeg, data.lngDeg,
                                                                      data.mean_latDeg_with_same_millisSinceGpsEpoch,
                                                                      data.mean_lngDeg_with_same_millisSinceGpsEpoch)

    data_processing(train_data)
    data_processing(test_data)

    def derived_data_processing(derived_data, data):
        # 添加correctedPrm特征
        derived_data["correctedPrm"] = derived_data["rawPrM"] + derived_data["satClkBiasM"] - \
                                       derived_data["isrbM"] - derived_data["ionoDelayM"] - \
                                       derived_data["tropoDelayM"]

        # 信号类型与卫星id
        derived_data["signalType_svid"] = derived_data["signalType"] + '_' + derived_data["svid"].astype("string")

        # 时间对齐
        data["millisSinceGpsEpoch/1000_round"] = np.round(data["millisSinceGpsEpoch"] / 1000).astype(np.int64)
        derived_data["millisSinceGpsEpoch/1000_round"] = np.round(derived_data["millisSinceGpsEpoch"] / 1000).astype(
            np.int64)

        # data和derived_data合并
        groups = derived_data.groupby(["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"])
        data = data.merge(
            groups["signalType_svid"].agg(lambda group: group.values),
            on=["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"], how="left"
        )
        data = data.merge(
            groups[["correctedPrm", "rawPrUncM", "satClkDriftMps"]].mean().rename(columns={
                "correctedPrm": "correctedPrm_avg",
                "rawPrUncM": "rawPrUncM_avg",
                "satClkDriftMps": "satClkDriftMps_avg"
            }),
            on=["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"], how="left"
        )
        data = data.merge(
            groups[["correctedPrm", "rawPrUncM", "satClkDriftMps"]].std().rename(columns={
                "correctedPrm": "correctedPrm_std",
                "rawPrUncM": "rawPrUncM_std",
                "satClkDriftMps": "satClkDriftMps_std"
            }),
            on=["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"], how="left"
        )

        return data

    train_data = derived_data_processing(train_derived_data, train_data)
    test_data = derived_data_processing(test_derived_data, test_data)

    # 分桶去null
    num_buckets = 20
    attr_names = ["correctedPrm", "rawPrUncM", "satClkDriftMps"]
    concat_data = pd.concat([train_data, test_data])
    for fn_name in ["avg", "std"]:
        for attr_name in attr_names:
            concat_data[attr_name + '_' + fn_name + "_bucketized"] = pd.qcut(concat_data[attr_name + '_' + fn_name],
                                                                      num_buckets, labels=False)
            concat_data.loc[concat_data[attr_name + '_' + fn_name + "_bucketized"].isnull(),
                     attr_name + '_' + fn_name + "_bucketized"] = num_buckets
            concat_data[attr_name + '_' + fn_name + "_bucketized"] = \
                concat_data[attr_name + '_' + fn_name + "_bucketized"].astype(np.int64)
            train_data[attr_name + '_' + fn_name + "_bucketized"] = \
                concat_data[:len(train_data)][attr_name + '_' + fn_name + "_bucketized"]
            test_data[attr_name + '_' + fn_name + "_bucketized"] = \
                concat_data[-len(test_data):][attr_name + '_' + fn_name + "_bucketized"]

    def raw_data_processing(raw_data, data):
        raw_data["millisSinceGpsEpoch"] = np.round((raw_data.TimeNanos - raw_data.FullBiasNanos) / 1000000).\
            astype(np.int64)
        raw_data["millisSinceGpsEpoch/1000_round"] = np.round(raw_data["millisSinceGpsEpoch"] / 1000).astype(np.int64)
        groups = raw_data.groupby(["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"])
        data = data.merge(
            groups[["Cn0DbHz", "BiasUncertaintyNanos"]].mean().rename(columns={
                "Cn0DbHz": "Cn0DbHz_avg",
                "BiasUncertaintyNanos": "BiasUncertaintyNanos_avg",
            }),
            on=["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"], how="left"
        )
        data = data.merge(
            groups[["Cn0DbHz", "BiasUncertaintyNanos"]].std().rename(columns={
                "Cn0DbHz": "Cn0DbHz_std",
                "BiasUncertaintyNanos": "BiasUncertaintyNanos_std",
            }),
            on=["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"], how="left"
        )
        data = data.merge(
            groups["utcTimeMillis"].first(),
            on=["collectionName", "phoneName", "millisSinceGpsEpoch/1000_round"], how="left"
        )
        return data

    train_data = raw_data_processing(train_raw_data, train_data)
    test_data = raw_data_processing(test_raw_data, test_data)

    def sensor_data_processing(gyro_data, mag_data, accel_data, data):
        data["utcTimeMillis/1000_round"] = np.round(data.utcTimeMillis / 1000).astype(np.int64)
        axises = ['X', 'Y', 'Z']
        prefix_suffix_df_dicts = [
            {"prefix": "UncalAccel", "suffix": "Mps2", "sensor_data": accel_data},
            {"prefix": "UncalGyro", "suffix": "RadPerSec", "sensor_data": gyro_data},
            {"prefix": "UncalMag", "suffix": "MicroT", "sensor_data": mag_data}
        ]
        for prefix_suffix_df_dict in prefix_suffix_df_dicts:
            prefix = prefix_suffix_df_dict["prefix"]
            suffix = prefix_suffix_df_dict["suffix"]
            sensor_data = prefix_suffix_df_dict["sensor_data"]
            sensor_data["utcTimeMillis"] = sensor_data.utcTimeMillis.astype(np.int64)
            sensor_data["utcTimeMillis/1000_round"] = np.round(sensor_data.utcTimeMillis / 1000).astype(np.int64)
            groups = sensor_data.groupby(["collectionName", "phoneName", "utcTimeMillis/1000_round"])
            data = data.merge(groups[[prefix + axis + suffix for axis in axises]].mean().rename(columns={
                prefix + axis + suffix: prefix + axis + suffix + "_avg" for axis in axises
            }), on=["collectionName", "phoneName", "utcTimeMillis/1000_round"], how="left")
            data = data.merge(groups[[prefix + axis + suffix for axis in axises]].std().rename(columns={
                prefix + axis + suffix: prefix + axis + suffix + "_std" for axis in axises
            }), on=["collectionName", "phoneName", "utcTimeMillis/1000_round"], how="left")

        col_names = [
            "UncalAccelXMps2_avg",
            "UncalAccelYMps2_avg",
            "UncalAccelZMps2_avg",
            "UncalAccelXMps2_std",
            "UncalAccelYMps2_std",
            "UncalAccelZMps2_std",
            "UncalGyroXRadPerSec_avg",
            "UncalGyroYRadPerSec_avg",
            "UncalGyroZRadPerSec_avg",
            "UncalGyroXRadPerSec_std",
            "UncalGyroYRadPerSec_std",
            "UncalGyroZRadPerSec_std",
            "UncalMagXMicroT_avg",
            "UncalMagYMicroT_avg",
            "UncalMagZMicroT_avg",
            "UncalMagXMicroT_std",
            "UncalMagYMicroT_std",
            "UncalMagZMicroT_std"
        ]
        groups = data.groupby(["collectionName", "phoneName"])
        data[[col_name + "+1" for col_name in col_names]] = groups[col_names].shift(-1)

        nan_idxs = data[data["UncalAccelXMps2_avg+1"].isnull()].index
        data.loc[nan_idxs, [col_name + "+1" for col_name in col_names]] = data.loc[nan_idxs, col_names].values

        data["ratio"] = (data["utcTimeMillis"] % 1000) / 1000
        data[col_names] = data[col_names].values * (1 - data["ratio"]).values.reshape(-1, 1) + \
            data[[col_name + "+1" for col_name in col_names]].values * data["ratio"].values.reshape(-1, 1)

        return data

    train_data = sensor_data_processing(train_gyro_data, train_mag_data, train_accel_data, train_data)
    test_data = sensor_data_processing(test_gyro_data, test_mag_data, test_accel_data, test_data)

    # 卡尔曼相关特征
    kf = KF()
    kf.apply_kf_smoothing(train_data, suffix="_kf")
    kf.apply_kf_smoothing(test_data, suffix="_kf")

    def add_kf_features(data):
        data["dist_between_base_and_kf"] = calc_haversine(data.latDeg, data.lngDeg, data.latDeg_kf, data.lngDeg_kf)
        data["base_kf_lat_delta"] = data["latDeg"] - data["latDeg_kf"]
        data["base_kf_lng_delta"] = data["lngDeg"] - data["lngDeg_kf"]
        data["abs_base_kf_lng_delta"] = abs(data["lngDeg"] - data["lngDeg_kf"])
        data["abs_base_kf_lat_delta"] = abs(data["latDeg"] - data["latDeg_kf"])

    add_kf_features(train_data)
    add_kf_features(test_data)

    # 缺失值均值填充
    col_names_ = [
        "UncalAccelXMps2_avg",
        "UncalAccelYMps2_avg",
        "UncalAccelZMps2_avg",
        "UncalAccelXMps2_std",
        "UncalAccelYMps2_std",
        "UncalAccelZMps2_std",
        "UncalGyroXRadPerSec_avg",
        "UncalGyroYRadPerSec_avg",
        "UncalGyroZRadPerSec_avg",
        "UncalGyroXRadPerSec_std",
        "UncalGyroYRadPerSec_std",
        "UncalGyroZRadPerSec_std",
        "UncalMagXMicroT_avg",
        "UncalMagYMicroT_avg",
        "UncalMagZMicroT_avg",
        "UncalMagXMicroT_std",
        "UncalMagYMicroT_std",
        "UncalMagZMicroT_std"
    ]
    train_data[col_names_] = train_data[col_names_].fillna(train_data[col_names_].mean())
    test_data[col_names_] = test_data[col_names_].fillna(test_data[col_names_].mean())

    # 切分训练验证
    train_pd = train_data[
        (train_data.collectionName == "2020-05-14-US-MTV-1") |
        (train_data.collectionName == "2020-05-21-US-MTV-2") |
        (train_data.collectionName == "2020-05-29-US-MTV-1") |
        (train_data.collectionName == "2020-06-04-US-MTV-1") |
        (train_data.collectionName == "2020-06-05-US-MTV-1") |
        (train_data.collectionName == "2020-07-08-US-MTV-1") |
        (train_data.collectionName == "2020-07-17-US-MTV-1") |
        (train_data.collectionName == "2020-08-06-US-MTV-2") |
        (train_data.collectionName == "2020-09-04-US-SF-1") |
        (train_data.collectionName == "2021-01-04-US-RWC-2") |
        (train_data.collectionName == "2021-01-05-US-SVL-1") |
        (train_data.collectionName == "2021-03-10-US-SVL-1") |
        (train_data.collectionName == "2021-04-22-US-SJC-1") |
        (train_data.collectionName == "2021-04-28-US-MTV-1") |
        (train_data.collectionName == "2021-04-29-US-SJC-2")
    ]
    val_pd = train_data[
        (train_data.collectionName == "2020-05-14-US-MTV-2") |
        (train_data.collectionName == "2020-05-21-US-MTV-1") |
        (train_data.collectionName == "2020-05-29-US-MTV-2") |
        (train_data.collectionName == "2020-06-05-US-MTV-2") |
        (train_data.collectionName == "2020-06-11-US-MTV-1") |
        (train_data.collectionName == "2020-07-17-US-MTV-2") |
        (train_data.collectionName == "2020-08-03-US-MTV-1") |
        (train_data.collectionName == "2020-09-04-US-SF-2") |
        (train_data.collectionName == "2021-01-04-US-RWC-1") |
        (train_data.collectionName == "2021-01-05-US-SVL-2") |
        (train_data.collectionName == "2021-04-15-US-MTV-1") |
        (train_data.collectionName == "2021-04-26-US-SVL-1") |
        (train_data.collectionName == "2021-04-28-US-SJC-1") |
        (train_data.collectionName == "2021-04-29-US-MTV-1")
    ]

    # 保存
    train_data.to_csv('traindata.csv', index=False)
    train_pd.to_csv("trainset_pandas.csv", index=False)
    val_pd.to_csv("valset_pandas.csv", index=False)
    test_data.to_csv("testset_pandas.csv", index=False)
