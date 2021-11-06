import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


MAX_DAYS_RANGE = [i for i in range(30,51)]
START_DENSITY_RANGE = [80, 85, 90]
DAY_STEP_RANGE = [i for i in range(5,11)]

DENSITY_STEP_RANGE = [i for i in range(1,8)]
MIN_DENSITY = 5
DENSITY_STEP_SIZE = 5

max_num_days = MAX_DAYS_RANGE[-1]
max_start_density = START_DENSITY_RANGE[-1]
max_density_step = DENSITY_STEP_RANGE[-1]
max_plant_cost = max_start_density * 0.12
max_num_spacing_changes = 9
max_spacing_cost = max_num_spacing_changes * 1.5 * max_num_days / 365
MAX_FEATURE = np.asarray(
    [max_start_density, max_num_days, max_plant_cost, max_spacing_cost, max_num_spacing_changes] + [max_density_step]*(max_num_days-1))

MAX_FEATURE_CVAE = np.asarray(
    [max_start_density, max_plant_cost, max_spacing_cost, max_num_spacing_changes] + [max_density_step]*(max_num_days-1))

min_start_density = START_DENSITY_RANGE[0]
min_num_days = MAX_DAYS_RANGE[0]
min_plant_cost = MIN_DENSITY * 0.12
min_num_spacing_changes = 2
min_spacing_cost = min_num_spacing_changes * 1.5 * min_num_days / 365
MIN_FEATURE = np.asarray([min_start_density, min_num_days, min_plant_cost, min_spacing_cost, min_num_spacing_changes] + [0] * (max_num_days-1))

MIN_FEATURE_CVAE = np.asarray([min_start_density, min_plant_cost, min_spacing_cost, min_num_spacing_changes] + [0] * (max_num_days-1))


GOOD_PD_STRS = [
    "1 80; 11 45; 19 25; 27 15",
    "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
    "1 80; 9 50; 14 25; 20 20; 27 15",
    "1 80; 12 45; 20 25; 27 20; 35 10",  # from email
    "1 80; 10 40; 20 30; 25 20; 30 10",  # from control sample
    "1 80; 10 55; 15 40; 20 25; 25 20; 31 15",  # from C15TEST
    "1 85; 7 50; 15 30; 25 20; 33 15",  # from D1 test on sim C
    "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
]


def normalize(feat, min_feat, max_feat):
    return (feat - min_feat) / (max_feat - min_feat)


def unnormalize(feat, min_feat, max_feat):
    return feat * (max_feat - min_feat) + min_feat


def sample_data(num_samples):
    print("samping plant density data ...")

    data = []
    for _ in tqdm(range(num_samples)):
        setpoints = []
        day, density = 1, random.choice(START_DENSITY_RANGE)
        max_days = random.choice(MAX_DAYS_RANGE)

        while day <= max_days and density >= MIN_DENSITY:
            setpoints.append(f'{day} {density}')
            day += random.choice(DAY_STEP_RANGE)
            density -= random.choice(DENSITY_STEP_RANGE) * DENSITY_STEP_SIZE

        pd_str = '; '.join(setpoints)
        data.append((max_days, pd_str))

    return data


def pd_str2feat(num_days, pd_str):
    pd_arr = pd_str2arr(max_num_days, pd_str)
    density_diff = (pd_arr[:-1] - pd_arr[1:]) / DENSITY_STEP_SIZE

    start_density = pd_arr[0]
    plant_cost = compute_plant_cost(num_days, pd_str)
    spacing_cost, num_spacing_changes = compute_spacing_cost(num_days, pd_str)
    key_features = np.asarray([
        start_density, num_days, plant_cost, spacing_cost, num_spacing_changes])

    feat = np.concatenate([key_features, density_diff], axis=0)
    return normalize(feat, MIN_FEATURE, MAX_FEATURE)


def pd_str2feat_cvae(num_days, pd_str):
    pd_arr = pd_str2arr(max_num_days, pd_str)
    density_diff = (pd_arr[:-1] - pd_arr[1:]) / DENSITY_STEP_SIZE

    start_density = pd_arr[0]
    plant_cost = compute_plant_cost(num_days, pd_str)
    spacing_cost, num_spacing_changes = compute_spacing_cost(num_days, pd_str)
    key_features = np.asarray([
        start_density, plant_cost, spacing_cost, num_spacing_changes])

    feat = np.concatenate([key_features, density_diff], axis=0)
    return normalize(feat, MIN_FEATURE_CVAE, MAX_FEATURE_CVAE)


def pd_str2arr(max_days, pd_str):
    arr = np.zeros(max_days, dtype=np.int32)
    setpoints = [sp_str.split() for sp_str in pd_str.split('; ')]
    for day, val in setpoints:
        day = int(day)
        val = int(val)
        arr[day-1:] = val
    return arr


def pd_feat2list(pd_feat, num_days=None):
    pd_feat = unnormalize(pd_feat, MIN_FEATURE, MAX_FEATURE)
    start_density = round(pd_feat[0])
    num_days = round(pd_feat[1]) if num_days is None else num_days
    pd_diff = pd_feat[5:5+num_days-1]

    pd_list = [start_density]
    pd = start_density
    for diff in pd_diff:
        pd -= round(diff) * DENSITY_STEP_SIZE
        pd_list.append(pd)
    
    return pd_list


def pd_feat2list_cvae(pd_feat, num_days):
    pd_feat = unnormalize(pd_feat, MIN_FEATURE_CVAE, MAX_FEATURE_CVAE)
    print(pd_feat)
    start_density = round(pd_feat[0])
    pd_diff = pd_feat[4:4+num_days-1]

    pd_list = [start_density]
    pd = start_density
    for diff in pd_diff:
        pd -= round(diff) * DENSITY_STEP_SIZE
        pd_list.append(pd)
    
    return pd_list


def compute_plant_cost(max_days, pd_str):
    pd_arr = pd_str2arr(max_days, pd_str)
    return 0.12 * compute_averageHeadPerM2(pd_arr)


def compute_spacing_cost(max_days, pd_str):
    num_spacing_changes = compute_num_spacing_changes(pd_str)
    fraction_of_year = max_days / 365
    return num_spacing_changes * 1.5 * fraction_of_year, num_spacing_changes


def compute_averageHeadPerM2(pd_arr):
    return len(pd_arr) / (1 / pd_arr).sum()


def compute_num_spacing_changes(pd_str):
    return len(pd_str.split('; ')) - 1


class PlantDensityDataset(Dataset):
    def __init__(self, num_samples, str2feat_func=pd_str2feat):
        super(PlantDensityDataset, self).__init__()
        self.num_samples = num_samples
        self.raw_data = sample_data(num_samples)
        self.str2feat = str2feat_func
        self.pd_features = self.transform(self.raw_data)
        
        min_days = MAX_DAYS_RANGE[0]
        max_days = MAX_DAYS_RANGE[-1]
        bandwith = max_days - min_days
        
        self.num_days = [ (t[0]-min_days)/bandwith for t in self.raw_data]

    def __getitem__(self, index):
        return self.pd_features[index], self.num_days[index]

    def __len__(self):
        return self.num_samples

    def transform(self, data):
        features = [self.str2feat(num_days, pd_str) for num_days, pd_str in data]
        return np.asarray(features, dtype=np.float32)

    @property
    def feature_dim(self):
        return len(self.pd_features[0])


if __name__ == '__main__':
    # dataset = PlantDensityDataset(1000, 50)
    # print(dataset.data)
    # print(dataset.data.shape)
    # print(dataset[-1])

    # for pd_scheme in sample_data(10):
    #     print(pd_scheme)

    # pd_str = "1 90; 8 65; 18 45; 26 25; 35 15"
    # pd_arr = pd_str2arr(65, pd_str)
    # print(compute_averageHeadPerM2(pd_arr))
    # print(compute_spacing_cost(65, pd_str))
    # print(compute_plant_cost(pd_arr))
    
    pd_str = "1 90; 8 65; 18 45; 26 25; 35 15"
    num_days = 65
    feat = pd_str2feat(num_days, pd_str)
    # print(feat.shape)

    # dataset = PlantDensityDataset(10)
    # for i in range(len(dataset)):
    #     print(i, dataset[i])
