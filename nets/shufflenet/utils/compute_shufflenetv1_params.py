def comput_unit_0(input_depth, num_groups, output_depth, num_bn_per_depth_params=2, first_stage_first_unit=False):
    """

    Args:
        input_depth:
        num_groups:
        output_depth:
        num_bn_per_depth_params:
        first_stage_first_unit:

    Returns:

    """
    bottleneck_depth = (output_depth) // 4

    if bottleneck_depth % num_groups != 0:
        bottleneck_depth = bottleneck_depth - bottleneck_depth % num_groups

    GConv1 = (input_depth // num_groups * bottleneck_depth // num_groups * num_groups
              ) if not first_stage_first_unit else (input_depth * bottleneck_depth)
    BN1 = bottleneck_depth * num_bn_per_depth_params
    DW = bottleneck_depth * 9
    BN2 = bottleneck_depth * num_bn_per_depth_params
    GConv2 = (bottleneck_depth // num_groups * (output_depth - input_depth) // num_groups * num_groups)
    BN3 = (output_depth - input_depth) * num_bn_per_depth_params
    num = GConv1 + BN1 + DW + BN2 + GConv2 + BN3
    return num


def comput_unit_not_0(input_depth, num_groups, num_bn_per_depth_params=2):
    """

    Args:
        input_depth:
        num_groups:
        num_bn_per_depth_params:

    Returns:

    """
    bottleneck_depth = input_depth // 4
    if bottleneck_depth % num_groups != 0:
        bottleneck_depth = bottleneck_depth - bottleneck_depth % num_groups
    GConv1 = (input_depth // num_groups * bottleneck_depth // num_groups * num_groups)
    BN1 = bottleneck_depth * num_bn_per_depth_params
    DW = bottleneck_depth * 9
    BN2 = bottleneck_depth * num_bn_per_depth_params
    GConv2 = (bottleneck_depth // num_groups * input_depth // num_groups * num_groups)
    BN3 = input_depth * num_bn_per_depth_params
    num = GConv1 + BN1 + DW + BN2 + GConv2 + BN3
    return num


endpoints_groups_0_25 = {
    '1': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 1215, 'Stage_0/Unit_1': 2106,
        'Stage_0/Unit_2': 2997, 'Stage_0/Unit_3': 3888,
        'Stage_1/Unit_0': 5562, 'Stage_1/Unit_1': 8640,
        'Stage_1/Unit_2': 11718, 'Stage_1/Unit_3': 14796,
        'Stage_1/Unit_4': 17874, 'Stage_1/Unit_5': 20952,
        'Stage_1/Unit_6': 24030, 'Stage_1/Unit_7': 27108,
        'Stage_2/Unit_0': 33048, 'Stage_2/Unit_1': 44388,
        'Stage_2/Unit_2': 55728, 'Stage_2/Unit_3': 67068
    },
    '2': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 1422, 'Stage_0/Unit_1': 2352,
        'Stage_0/Unit_2': 3282, 'Stage_0/Unit_3': 4212,
        'Stage_1/Unit_0': 5922, 'Stage_1/Unit_1': 8982,
        'Stage_1/Unit_2': 12042, 'Stage_1/Unit_3': 15102,
        'Stage_1/Unit_4': 18162, 'Stage_1/Unit_5': 21222,
        'Stage_1/Unit_6': 24282, 'Stage_1/Unit_7': 27342,
        'Stage_2/Unit_0': 33392, 'Stage_2/Unit_1': 44742,
        'Stage_2/Unit_2': 56092, 'Stage_2/Unit_3': 67442
    },
    '3': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 1593, 'Stage_0/Unit_1': 2598,
        'Stage_0/Unit_2': 3603, 'Stage_0/Unit_3': 4608,
        'Stage_1/Unit_0': 6438, 'Stage_1/Unit_1': 9648,
        'Stage_1/Unit_2': 12858, 'Stage_1/Unit_3': 16068,
        'Stage_1/Unit_4': 19278, 'Stage_1/Unit_5': 22488,
        'Stage_1/Unit_6': 25698, 'Stage_1/Unit_7': 28908,
        'Stage_2/Unit_0': 34968, 'Stage_2/Unit_1': 46188,
        'Stage_2/Unit_2': 57408, 'Stage_2/Unit_3': 68628
    },
    '4': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 1652, 'Stage_0/Unit_1': 2640,
        'Stage_0/Unit_2': 3628, 'Stage_0/Unit_3': 4616,
        'Stage_1/Unit_0': 6388, 'Stage_1/Unit_1': 9452,
        'Stage_1/Unit_2': 12516, 'Stage_1/Unit_3': 15580,
        'Stage_1/Unit_4': 18644, 'Stage_1/Unit_5': 21708,
        'Stage_1/Unit_6': 24772, 'Stage_1/Unit_7': 27836,
        'Stage_2/Unit_0': 33888, 'Stage_2/Unit_1': 44972,
        'Stage_2/Unit_2': 56056, 'Stage_2/Unit_3': 67140
    },
    '8': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 2088, 'Stage_0/Unit_1': 3312,
        'Stage_0/Unit_2': 4536, 'Stage_0/Unit_3': 5760,
        'Stage_1/Unit_0': 7920, 'Stage_1/Unit_1': 11520,
        'Stage_1/Unit_2': 15120, 'Stage_1/Unit_3': 18720,
        'Stage_1/Unit_4': 22320, 'Stage_1/Unit_5': 25920,
        'Stage_1/Unit_6': 29520, 'Stage_1/Unit_7': 33120,
        'Stage_2/Unit_0': 39744, 'Stage_2/Unit_1': 51552,
        'Stage_2/Unit_2': 63360, 'Stage_2/Unit_3': 75168
    },
}

endpoints_groups_0_5 = {
    '1': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 2430, 'Stage_0/Unit_1': 5508,
        'Stage_0/Unit_2': 8586, 'Stage_0/Unit_3': 11664,
        'Stage_1/Unit_0': 17604, 'Stage_1/Unit_1': 28944,
        'Stage_1/Unit_2': 40284, 'Stage_1/Unit_3': 51624,
        'Stage_1/Unit_4': 62964, 'Stage_1/Unit_5': 74304,
        'Stage_1/Unit_6': 85644, 'Stage_1/Unit_7': 96984,
        'Stage_2/Unit_0': 119232, 'Stage_2/Unit_1': 162648,
        'Stage_2/Unit_2': 206064, 'Stage_2/Unit_3': 249480
    },
    '2': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 2796, 'Stage_0/Unit_1': 5856,
        'Stage_0/Unit_2': 8916, 'Stage_0/Unit_3': 11976,
        'Stage_1/Unit_0': 18026, 'Stage_1/Unit_1': 29376,
        'Stage_1/Unit_2': 40726, 'Stage_1/Unit_3': 52076,
        'Stage_1/Unit_4': 63426, 'Stage_1/Unit_5': 74776,
        'Stage_1/Unit_6': 86126, 'Stage_1/Unit_7': 97476,
        'Stage_2/Unit_0': 119576, 'Stage_2/Unit_1': 162276,
        'Stage_2/Unit_2': 204976, 'Stage_2/Unit_3': 247676
    },
    '3': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 3138, 'Stage_0/Unit_1': 6348,
        'Stage_0/Unit_2': 9558, 'Stage_0/Unit_3': 12768,
        'Stage_1/Unit_0': 18828, 'Stage_1/Unit_1': 30048,
        'Stage_1/Unit_2': 41268, 'Stage_1/Unit_3': 52488,
        'Stage_1/Unit_4': 63708, 'Stage_1/Unit_5': 74928,
        'Stage_1/Unit_6': 86148, 'Stage_1/Unit_7': 97368,
        'Stage_2/Unit_0': 119088, 'Stage_2/Unit_1': 160728,
        'Stage_2/Unit_2': 202368, 'Stage_2/Unit_3': 244008
    },
    '4': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 3200, 'Stage_0/Unit_1': 6264,
        'Stage_0/Unit_2': 9328, 'Stage_0/Unit_3': 12392,
        'Stage_1/Unit_0': 18444, 'Stage_1/Unit_1': 29528,
        'Stage_1/Unit_2': 40612, 'Stage_1/Unit_3': 51696,
        'Stage_1/Unit_4': 62780, 'Stage_1/Unit_5': 73864,
        'Stage_1/Unit_6': 84948, 'Stage_1/Unit_7': 96032,
        'Stage_2/Unit_0': 117384, 'Stage_2/Unit_1': 158048,
        'Stage_2/Unit_2': 198712, 'Stage_2/Unit_3': 239376
    },
    '8': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 4104, 'Stage_0/Unit_1': 7704,
        'Stage_0/Unit_2': 11304, 'Stage_0/Unit_3': 14904,
        'Stage_1/Unit_0': 21528, 'Stage_1/Unit_1': 33336,
        'Stage_1/Unit_2': 45144, 'Stage_1/Unit_3': 56952,
        'Stage_1/Unit_4': 68760, 'Stage_1/Unit_5': 80568,
        'Stage_1/Unit_6': 92376, 'Stage_1/Unit_7': 104184,
        'Stage_2/Unit_0': 126648, 'Stage_2/Unit_1': 168696,
        'Stage_2/Unit_2': 210744, 'Stage_2/Unit_3': 252792
    },
}

endpoints_groups = {
    '1': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 6804, 'Stage_0/Unit_1': 18144,
        'Stage_0/Unit_2': 29484, 'Stage_0/Unit_3': 40824,
        'Stage_1/Unit_0': 63072, 'Stage_1/Unit_1': 106488,
        'Stage_1/Unit_2': 149904, 'Stage_1/Unit_3': 193320,
        'Stage_1/Unit_4': 236736, 'Stage_1/Unit_5': 280152,
        'Stage_1/Unit_6': 323568, 'Stage_1/Unit_7': 366984,
        'Stage_2/Unit_0': 452952, 'Stage_2/Unit_1': 622728,
        'Stage_2/Unit_2': 792504, 'Stage_2/Unit_3': 962280
    },
    '2': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 7598, 'Stage_0/Unit_1': 18948,
        'Stage_0/Unit_2': 30298, 'Stage_0/Unit_3': 41648,
        'Stage_1/Unit_0': 63748, 'Stage_1/Unit_1': 106448,
        'Stage_1/Unit_2': 149148, 'Stage_1/Unit_3': 191848,
        'Stage_1/Unit_4': 234548, 'Stage_1/Unit_5': 277248,
        'Stage_1/Unit_6': 319948, 'Stage_1/Unit_7': 362648,
        'Stage_2/Unit_0': 446848, 'Stage_2/Unit_1': 612248,
        'Stage_2/Unit_2': 777648, 'Stage_2/Unit_3': 943048
    },
    '3': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 8028, 'Stage_0/Unit_1': 19248,
        'Stage_0/Unit_2': 30468, 'Stage_0/Unit_3': 41688,
        'Stage_1/Unit_0': 63408, 'Stage_1/Unit_1': 105048,
        'Stage_1/Unit_2': 146688, 'Stage_1/Unit_3': 188328,
        'Stage_1/Unit_4': 229968, 'Stage_1/Unit_5': 271608,
        'Stage_1/Unit_6': 313248, 'Stage_1/Unit_7': 354888,
        'Stage_2/Unit_0': 436728, 'Stage_2/Unit_1': 596808,
        'Stage_2/Unit_2': 756888, 'Stage_2/Unit_3': 916968
    },
    '4': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 8332, 'Stage_0/Unit_1': 19416,
        'Stage_0/Unit_2': 30500, 'Stage_0/Unit_3': 41584,
        'Stage_1/Unit_0': 62936, 'Stage_1/Unit_1': 103600,
        'Stage_1/Unit_2': 144264, 'Stage_1/Unit_3': 184928,
        'Stage_1/Unit_4': 225592, 'Stage_1/Unit_5': 266256,
        'Stage_1/Unit_6': 306920, 'Stage_1/Unit_7': 347584,
        'Stage_2/Unit_0': 427280, 'Stage_2/Unit_1': 582592,
        'Stage_2/Unit_2': 737904, 'Stage_2/Unit_3': 893216
    },
    '8': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 9864, 'Stage_0/Unit_1': 21672,
        'Stage_0/Unit_2': 33480, 'Stage_0/Unit_3': 45288,
        'Stage_1/Unit_0': 67752, 'Stage_1/Unit_1': 109800,
        'Stage_1/Unit_2': 151848, 'Stage_1/Unit_3': 193896,
        'Stage_1/Unit_4': 235944, 'Stage_1/Unit_5': 277992,
        'Stage_1/Unit_6': 320040, 'Stage_1/Unit_7': 362088,
        'Stage_2/Unit_0': 443880, 'Stage_2/Unit_1': 601704,
        'Stage_2/Unit_2': 759528, 'Stage_2/Unit_3': 917352
    },
}

endpoints_groups_1_5 = {
    '1': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 13770, 'Stage_0/Unit_1': 38556,
        'Stage_0/Unit_2': 63342, 'Stage_0/Unit_3': 88128,
        'Stage_1/Unit_0': 137052, 'Stage_1/Unit_1': 233280,
        'Stage_1/Unit_2': 329508, 'Stage_1/Unit_3': 425736,
        'Stage_1/Unit_4': 521964, 'Stage_1/Unit_5': 618192,
        'Stage_1/Unit_6': 714420, 'Stage_1/Unit_7': 810648,
        'Stage_2/Unit_0': 1001808, 'Stage_2/Unit_1': 1380888,
        'Stage_2/Unit_2': 1759968, 'Stage_2/Unit_3': 2139048
    },
    '2': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 14646, 'Stage_0/Unit_1': 38856,
        'Stage_0/Unit_2': 63066, 'Stage_0/Unit_3': 87276,
        'Stage_1/Unit_0': 135426, 'Stage_1/Unit_1': 229476,
        'Stage_1/Unit_2': 323526, 'Stage_1/Unit_3': 417576,
        'Stage_1/Unit_4': 511626, 'Stage_1/Unit_5': 605676,
        'Stage_1/Unit_6': 699726, 'Stage_1/Unit_7': 793776,
        'Stage_2/Unit_0': 980076, 'Stage_2/Unit_1': 1348176,
        'Stage_2/Unit_2': 1716276, 'Stage_2/Unit_3': 2084376
    },
    '3': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 15318, 'Stage_0/Unit_1': 39348,
        'Stage_0/Unit_2': 63378, 'Stage_0/Unit_3': 87408,
        'Stage_1/Unit_0': 134388, 'Stage_1/Unit_1': 225648,
        'Stage_1/Unit_2': 316908, 'Stage_1/Unit_3': 408168,
        'Stage_1/Unit_4': 499428, 'Stage_1/Unit_5': 590688,
        'Stage_1/Unit_6': 681948, 'Stage_1/Unit_7': 773208,
        'Stage_2/Unit_0': 953568, 'Stage_2/Unit_1': 1308888,
        'Stage_2/Unit_2': 1664208, 'Stage_2/Unit_3': 2019528
    },
    '4': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 15372, 'Stage_0/Unit_1': 38496,
        'Stage_0/Unit_2': 61620, 'Stage_0/Unit_3': 84744,
        'Stage_1/Unit_0': 130644, 'Stage_1/Unit_1': 219384,
        'Stage_1/Unit_2': 308124, 'Stage_1/Unit_3': 396864,
        'Stage_1/Unit_4': 485604, 'Stage_1/Unit_5': 574344,
        'Stage_1/Unit_6': 663084, 'Stage_1/Unit_7': 751824,
        'Stage_2/Unit_0': 926856, 'Stage_2/Unit_1': 1270800,
        'Stage_2/Unit_2': 1614744, 'Stage_2/Unit_3': 1958688
    },
    '8': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 17928, 'Stage_0/Unit_1': 42552,
        'Stage_0/Unit_2': 67176, 'Stage_0/Unit_3': 91800,
        'Stage_1/Unit_0': 139320, 'Stage_1/Unit_1': 230040,
        'Stage_1/Unit_2': 320760, 'Stage_1/Unit_3': 411480,
        'Stage_1/Unit_4': 502200, 'Stage_1/Unit_5': 592920,
        'Stage_1/Unit_6': 683640, 'Stage_1/Unit_7': 774360,
        'Stage_2/Unit_0': 952344, 'Stage_2/Unit_1': 1299672,
        'Stage_2/Unit_2': 1647000, 'Stage_2/Unit_3': 1994328
    },
}

endpoints_groups_2_0 = {
    '1': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 23328, 'Stage_0/Unit_1': 66744,
        'Stage_0/Unit_2': 110160, 'Stage_0/Unit_3': 153576,
        'Stage_1/Unit_0': 239544, 'Stage_1/Unit_1': 409320,
        'Stage_1/Unit_2': 579096, 'Stage_1/Unit_3': 748872,
        'Stage_1/Unit_4': 918648, 'Stage_1/Unit_5': 1088424,
        'Stage_1/Unit_6': 1258200, 'Stage_1/Unit_7': 1427976,
        'Stage_2/Unit_0': 1765800, 'Stage_2/Unit_1': 2437128,
        'Stage_2/Unit_2': 3108456, 'Stage_2/Unit_3': 3779784
    },
    '2': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 24548, 'Stage_0/Unit_1': 67248,
        'Stage_0/Unit_2': 109948, 'Stage_0/Unit_3': 152648,
        'Stage_1/Unit_0': 236848, 'Stage_1/Unit_1': 402248,
        'Stage_1/Unit_2': 567648, 'Stage_1/Unit_3': 733048,
        'Stage_1/Unit_4': 898448, 'Stage_1/Unit_5': 1063848,
        'Stage_1/Unit_6': 1229248, 'Stage_1/Unit_7': 1394648,
        'Stage_2/Unit_0': 1723048, 'Stage_2/Unit_1': 2373848,
        'Stage_2/Unit_2': 3024648, 'Stage_2/Unit_3': 3675448
    },
    '3': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 25008, 'Stage_0/Unit_1': 66648,
        'Stage_0/Unit_2': 108288, 'Stage_0/Unit_3': 149928,
        'Stage_1/Unit_0': 231768, 'Stage_1/Unit_1': 391848,
        'Stage_1/Unit_2': 551928, 'Stage_1/Unit_3': 712008,
        'Stage_1/Unit_4': 872088, 'Stage_1/Unit_5': 1032168,
        'Stage_1/Unit_6': 1192248, 'Stage_1/Unit_7': 1352328,
        'Stage_2/Unit_0': 1669608, 'Stage_2/Unit_1': 2296968,
        'Stage_2/Unit_2': 2924328, 'Stage_2/Unit_3': 3551688
    },
    '4': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 25264, 'Stage_0/Unit_1': 65928,
        'Stage_0/Unit_2': 106592, 'Stage_0/Unit_3': 147256,
        'Stage_1/Unit_0': 226952, 'Stage_1/Unit_1': 382264,
        'Stage_1/Unit_2': 537576, 'Stage_1/Unit_3': 692888,
        'Stage_1/Unit_4': 848200, 'Stage_1/Unit_5': 1003512,
        'Stage_1/Unit_6': 1158824, 'Stage_1/Unit_7': 1314136,
        'Stage_2/Unit_0': 1621496, 'Stage_2/Unit_1': 2228056,
        'Stage_2/Unit_2': 2834616, 'Stage_2/Unit_3': 3441176
    },
    '8': {
        'Conv2d_0': 720, 'MaxPool2d_0': 720,
        'Stage_0/Unit_0': 28296, 'Stage_0/Unit_1': 70344,
        'Stage_0/Unit_2': 112392, 'Stage_0/Unit_3': 154440,
        'Stage_1/Unit_0': 236232, 'Stage_1/Unit_1': 394056,
        'Stage_1/Unit_2': 551880, 'Stage_1/Unit_3': 709704,
        'Stage_1/Unit_4': 867528, 'Stage_1/Unit_5': 1025352,
        'Stage_1/Unit_6': 1183176, 'Stage_1/Unit_7': 1341000,
        'Stage_2/Unit_0': 1652040, 'Stage_2/Unit_1': 2262600,
        'Stage_2/Unit_2': 2873160, 'Stage_2/Unit_3': 3483720
    },
}

if __name__ == '__main__':
    for rate in [0.25, 0.5, 1, 1.5, 2]:
        DEPTH_CHANNELS_DEFS = {
            '1': [144, 288, 576],
            '2': [200, 400, 800],
            '3': [240, 480, 960],
            '4': [272, 544, 1088],
            '8': [384, 768, 1536],
        }

        print('*** ', rate, ' ***')
        for k in DEPTH_CHANNELS_DEFS:
            DEPTH_CHANNELS_DEFS[k] = [int(rate * depth) for depth in DEPTH_CHANNELS_DEFS[k]]

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
        ]

        for num_groups in DEPTH_CHANNELS_DEFS:
            params = [720, 720]
            depths = DEPTH_CHANNELS_DEFS[num_groups]
            num_groups = int(num_groups)

            for i in range(4):
                if not i:
                    params.append(comput_unit_0(24, num_groups, depths[0], first_stage_first_unit=True) + params[-1])
                else:
                    params.append(comput_unit_not_0(depths[0], num_groups) + params[-1])

            for i in range(8):
                if not i:
                    params.append(comput_unit_0(depths[0], num_groups, depths[1]) + params[-1])
                else:
                    params.append(comput_unit_not_0(depths[1], num_groups) + params[-1])

            for i in range(4):
                if not i:
                    params.append(comput_unit_0(depths[1], num_groups, depths[2]) + params[-1])
                else:
                    params.append(comput_unit_not_0(depths[2], num_groups) + params[-1])

            ret = {n: num for n, num in zip(endpoints, params)}
            print("'%d':" % num_groups, str(ret), ',')
        print()
