from easydict import EasyDict

def t1_mapping():
    extend_config = []

    base_link = "Trunk"
    joint_matches = [
        ["Waist", "Pelvis"],
        ["Hip_Yaw_Left", "L_Hip"],
        ["Shank_Left", "L_Knee"],
        ["Ankle_Cross_Left", "L_Ankle"],
        ["Hip_Yaw_Right", "R_Hip"],
        ["Shank_Right", "R_Knee"],
        ["Ankle_Cross_Right", "R_Ankle"],
        ["AL2", "L_Shoulder"],
        ["left_hand_link", "L_Elbow"],
        ["left_hand_link_2", "L_Hand"],
        ["AR2", "R_Shoulder"],
        ["right_hand_link", "R_Elbow"],
        ["right_hand_link_2", "R_Hand"],
        ["head", "Head"],
    ]
    # joint_matches = [
    #     ["Waist", "Pelvis"],
    #     ["Hip_Yaw_Left", "L_Hip"],
    #     ["Shank_Left", "L_Knee"],
    #     ["Ankle_Cross_Left", "L_Ankle"],
    #     ["Hip_Yaw_Right", "R_Hip"],
    #     ["Shank_Right", "R_Knee"],
    #     ["Ankle_Cross_Right", "R_Ankle"],
    #     ["AL1", "L_Shoulder"],
    #     ["AL3", "L_Elbow"],
    #     ["left_hand_link", "L_Hand"],
    #     ["AR1", "R_Shoulder"],
    #     ["AR3", "R_Elbow"],
    #     ["right_hand_link", "R_Hand"],
    #     ["H2", "Head"],
    # ]

    smpl_pose_modifier = []

    asset_file = "../description/robots/t1/T1_Serial.mjcf"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )


def t1_mapping_with_extend_config():
    extend_config = [
        {
            "joint_name": "left_hand_link_2",
            "parent_name": "left_hand_link",
            "pos": [0.0, 0.20, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_hand_link_2",
            "parent_name": "right_hand_link",
            "pos": [0.0, -0.20, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "head",
            "parent_name": "H2",
            "pos": [0.0, 0.0, 0.15],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "left_toe_link",
            "parent_name": "left_foot_link",
            "pos": [0.08, 0.0, -0.01],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_toe_link",
            "parent_name": "right_foot_link",
            "pos": [0.08, 0.0, -0.01],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
    ]

    base_link = "Trunk"
    joint_matches = [
        ["Waist", "Pelvis"],
        ["Hip_Yaw_Left", "L_Hip"],
        ["Shank_Left", "L_Knee"],
        ["Ankle_Cross_Left", "L_Ankle"],
        ["Hip_Yaw_Right", "R_Hip"],
        ["Shank_Right", "R_Knee"],
        ["Ankle_Cross_Right", "R_Ankle"],
        ["AL2", "L_Shoulder"],
        ["left_hand_link", "L_Elbow"],
        ["left_hand_link_2", "L_Hand"],
        ["AR2", "R_Shoulder"],
        ["right_hand_link", "R_Elbow"],
        ["right_hand_link_2", "R_Hand"],
        ["head", "Head"],
    ]

    smpl_pose_modifier = []

    asset_file = "../description/robots/t1/T1_Serial.mjcf"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )


def g1_mapping():
    #### Config for extension
    extend_config = [
        {
            "joint_name": "left_rubber_hand_2",
            "parent_name": "left_wrist_yaw_link",
            "pos": [0.12, 0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_rubber_hand_2",
            "parent_name": "right_wrist_yaw_link",
            "pos": [0.12, 0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "head",
            "parent_name": "pelvis",
            "pos": [0.0, 0.0, 0.4],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
                {
            "joint_name": "left_toe_link",
            "parent_name": "left_ankle_roll_link",
            "pos": [0.08, 0.0, -0.01],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_toe_link",
            "parent_name": "right_ankle_roll_link",
            "pos": [0.08, 0.0, -0.01],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        
    ]

    base_link = "torso_link"
    joint_matches = [
        ["pelvis", "Pelvis"],
        ["left_hip_yaw_link", "L_Hip"],
        ["left_knee_link", "L_Knee"],
        ["left_ankle_roll_link", "L_Ankle"],
        ["right_hip_yaw_link", "R_Hip"],
        ["right_knee_link", "R_Knee"],
        ["right_ankle_roll_link", "R_Ankle"],
        ["left_shoulder_pitch_link", "L_Shoulder"],
        ["left_elbow_link", "L_Elbow"],
        ["left_rubber_hand_2", "L_Hand"],
        ["right_shoulder_pitch_link", "R_Shoulder"],
        ["right_elbow_link", "R_Elbow"],
        ["right_rubber_hand_2", "R_Hand"],
        ["head", "Head"],
    ]

    smpl_pose_modifier = [
        {"Pelvis": "[np.pi/2, 0, np.pi/2]"},
        {"L_Shoulder": "[0, 0, -np.pi/2]"},
        {"R_Shoulder": "[0, 0, np.pi/2]"},
        {"L_Elbow": "[0, -np.pi/2, 0]"},
        {"R_Elbow": "[0, np.pi/2, 0]"},
    ]

    asset_file = "../description/robots/g1/g1_29dof_rev_1_0.xml"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )

def smplx_with_limits_mapping():
    #### Config for extension
    extend_config = []

    base_link = "Pelvis"

    smplx_joints = [
        "Pelvis",
        "L_Hip",
        "L_Knee",
        "L_Ankle",
        "L_Toe",
        "R_Hip",
        "R_Knee",
        "R_Ankle",
        "R_Toe",
        "Torso",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "L_Thorax",
        "L_Shoulder",
        "L_Elbow",
        "L_Wrist",
        "L_Index1",
        "L_Index2",
        "L_Index3",
        "L_Middle1",
        "L_Middle2",
        "L_Middle3",
        "L_Pinky1",
        "L_Pinky2",
        "L_Pinky3",
        "L_Ring1",
        "L_Ring2",
        "L_Ring3",
        "L_Thumb1",
        "L_Thumb2",
        "L_Thumb3",
        "R_Thorax",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "R_Index1",
        "R_Index2",
        "R_Index3",
        "R_Middle1",
        "R_Middle2",
        "R_Middle3",
        "R_Pinky1",
        "R_Pinky2",
        "R_Pinky3",
        "R_Ring1",
        "R_Ring2",
        "R_Ring3",
        "R_Thumb1",
        "R_Thumb2",
        "R_Thumb3",
    ]
    joint_matches = [[joint, joint] for joint in smplx_joints]

    smpl_pose_modifier = []

    asset_file = "data/assets/mjcf/smplx_humanoid_with_limits.xml"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )


def get_config(humanoid_type: str):
    if humanoid_type == "g1":
        return g1_mapping()
    elif humanoid_type == "t1":
        # return t1_mapping()
        return t1_mapping_with_extend_config()
    elif humanoid_type == "smplx_humanoid_with_limits":
        return smplx_with_limits_mapping()
    else:
        raise NotImplementedError
