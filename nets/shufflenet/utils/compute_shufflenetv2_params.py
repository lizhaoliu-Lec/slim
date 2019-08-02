# how many params for a certain feature map of BN
BN_FACTOR = 2

# if True, then compute params sum up to the scope
# else, compute params exactly for the scope
UP_TO_SCOPE = True


def compute_conv2d_0(params, channels_out):
    params.append(3 * 3 * 3 * channels_out + channels_out * BN_FACTOR)
    params.append(params[-1])
    return params


def compute_unit_0(channels_in, channels_out):
    branch_out = channels_out / 2
    left_DWConv = 3 * 3 * channels_in
    BN = channels_in * BN_FACTOR
    left_1x1Conv = channels_in * branch_out
    BN += branch_out * BN_FACTOR

    right_1x1Conv_1 = channels_in * branch_out
    right_DWConv = 3 * 3 * branch_out
    right_1x1Conv_2 = branch_out * branch_out
    BN += branch_out * BN_FACTOR * 3
    return int(left_DWConv + left_1x1Conv +
               right_1x1Conv_1 + right_DWConv + right_1x1Conv_2 + BN)


def compute_unit_not_0(channels_out):
    branch_out = channels_out / 2
    Conv1x1_1 = branch_out * branch_out
    DWConv = 3 * 3 * branch_out
    Conv1x1_2 = branch_out * branch_out
    BN = branch_out * BN_FACTOR * 3
    return int(Conv1x1_1 + DWConv + Conv1x1_2 + BN)


def compute_conv5(params, channels_in, channels_out):
    if UP_TO_SCOPE:
        params.append(params[-1] + int(channels_in * channels_out + channels_out * BN_FACTOR))
    else:
        params.append(int(channels_in * channels_out + channels_out * BN_FACTOR))
    return params


def compute_stage(params, channels_in, channels_out, num_units):
    for i in range(num_units):
        if i == 0:
            if UP_TO_SCOPE:
                params.append(params[-1] + compute_unit_0(channels_in, channels_out))
            else:
                params.append(compute_unit_0(channels_in, channels_out))
        else:
            if UP_TO_SCOPE:
                params.append(params[-1] + compute_unit_not_0(channels_out))
            else:
                params.append(compute_unit_not_0(channels_out))
    return params


endpoints = [
    'Conv2d_0',
    'MaxPool2d_0',
    # Stage 1 with 4
    'Stage_0/Unit_0',
    'Stage_0/Unit_1',
    'Stage_0/Unit_2',
    'Stage_0/Unit_3',
    # Stage 2 with 8,
    'Stage_1/Unit_0',
    'Stage_1/Unit_1',
    'Stage_1/Unit_2',
    'Stage_1/Unit_3',
    'Stage_1/Unit_4',
    'Stage_1/Unit_5',
    'Stage_1/Unit_6',
    'Stage_1/Unit_7',
    # Stage 3 with 4,
    'Stage_2/Unit_0',
    'Stage_2/Unit_1',
    'Stage_2/Unit_2',
    'Stage_2/Unit_3',
    'Conv2d_1'
]

complicity = {
    '0.5': [24, 48, 96, 192, 1024],
    '1.0': [24, 116, 232, 464, 1024],
    '1.5': [24, 176, 352, 704, 1024],
    '2.0': [24, 244, 488, 976, 2048],
}

if __name__ == "__main__":
    for i in complicity.keys():
        params = []
        params = compute_conv2d_0(params, complicity[i][0])
        params = compute_stage(params, complicity[i][0], complicity[i][1], 4)
        params = compute_stage(params, complicity[i][1], complicity[i][2], 8)
        params = compute_stage(params, complicity[i][2], complicity[i][3], 4)
        params = compute_conv5(params, complicity[i][3], complicity[i][4])
        params_dict = {sc: para for sc, para in zip(endpoints, params)}
        print("-" * 80)
        print("ShufflenetV2 complicity: ", i)
        for j in params_dict.keys():
            print('\'' + j + '\'' + ': ' + str(params_dict[j]) + ',')
        print("-" * 80)
