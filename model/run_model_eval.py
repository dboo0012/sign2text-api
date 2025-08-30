import time
import torch
import os
import json
import numpy as np
from transformers import AutoTokenizer
from utils.easydict import EasyDict as edict
from train_how2_pose_DDP_inter_VN import How2SignPoseDataset, construct_model, eval
import json
from tqdm import tqdm
import concurrent.futures



def process_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    pose_data = []
    for person in data.get('people', []):
        # Extract keypoints in the specified order
        for key in ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', 'face_keypoints_2d']:
            if key in person:
                pose_data.extend(person[key])

    # Reshape to (vertices, channels)
    pose_array = np.array(pose_data).reshape(-1, 3)
    return pose_array

def process_subfolder(subfolder_path):
    json_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.json')])
    frame_data = []

    for json_file in tqdm(json_files, desc=f"Processing {subfolder_path}"):
        json_path = os.path.join(subfolder_path, json_file)
        frame_data.append(process_json_file(json_path))

    # Stack frames along the time axis: (Time, Vertices, Channels)
    frame_data = np.stack(frame_data)

    return frame_data


def run_inference(input_data):
    # --------- CONFIGURATION ---------
    work_dir_prefix = r"C:\Monash\Y3_SEM-1\FYP_2\Alt\Glofe\GloFE-main"
    work_dir = "how2sign/vn_model"
    tokenizer_path = "notebooks/how2sign/how2sign-bpe25000-tokenizer-uncased"
    weights_path = os.path.join(work_dir_prefix, "how2sign", "vn_model", "glofe_vn_how2sign_0224.pt")
    label_path = "how2sign/dataset/how2sign_realigned_val.csv"
    feat_path = None  # Set if needed

    args_dict = {
        'bs': 1,  # batch size for single inference
        'local_rank': 0,
        'ngpus': 1,
        'clip_length': 16,
        'prefix_dim': 2048,
        'eos_token': '.',
        'tokenizer': tokenizer_path,
        'feat_path': feat_path,
        'label_path': label_path,
        'max_gen_tks': 35,
        'num_beams': 5,
        'work_dir_prefix': work_dir_prefix,
        'work_dir': work_dir,
        'phase': 'test',
        # Add other required args here
    }

    # --------- LOAD TOKENIZER ---------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    # --------- LOAD MODEL CONFIG ---------
    output_dir = os.path.join(work_dir_prefix, work_dir)
    config_path = os.path.join(output_dir, "exp_config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path, 'r'))
    else:
        config = args_dict

    # --------- LOAD MODEL ---------
    from models.trans_model_inter_vn import TransBaseModel
    model_cls = TransBaseModel
    model = construct_model(model_cls, edict(config), distributed=False)
    weights = torch.load(weights_path, map_location='cpu')

    # Remove 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # --------- LOAD DATASET ---------
    dataset = How2SignPoseDataset(config, tokenizer=tokenizer, phase='test', split='test')


    input_folder = "how2sign\dataset\openpose_output\json"


    subfolders = [os.path.join(input_folder, d) for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]


    # Start timer for evaluation
    eval_start_time = time.time()




    # Load the json to Joints directly

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor for faster I/O-bound operations
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_subfolder, subfolder) for subfolder in subfolders]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    print("Joints " \
    "shape")

    joints = results[0]
    # print(joints.shape)  # (126, 137, 3)
    # print(joints[0].shape) # (137, 3)
    # print(joints[1].shape) # (137, 3)
    # print(joints[2].shape) # (137, 3)







    # read files

    # vid_index = 7
    # vid_name = input_data["keypoints_name"]

    # file_path = os.path.join("how2sign\dataset\openpose_output\pkl", f'{vid_name}.pkl')  # Change data input name here

    # print(f'\n\nReading pose file: {file_path}')

    # # try:
    # with open(file_path, 'rb') as f:
    #     joints = pickle.load(f)


    # Preprocess keypoints into valid input shape







    # print(f'Pose data loaded. Shape: {[j.shape for j in joints]}')  # Print shapes of the loaded arrays


    # Save joints[0], joints[1], joints[2] to separate JSON files
    # with open('num_frame_json.json', 'w') as f:
    #     json.dump(joints[0].tolist(), f, indent=2)
    # with open('keypoints.json', 'w') as f:
    #     json.dump(joints[1].tolist(), f, indent=2)
    # with open('coord_json.json', 'w') as f:
    #     json.dump(joints[2].tolist(), f, indent=2)


        

    # # Start timer for evaluation
    # eval_start_time = time.time()

    # -------------------------------------Initialise Keypoint Preprocessing Config --------------------------------------------
    # Openpose 78
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    body_sample_indices = [x for x in range(25) if x not in body_pose_exclude]

    face_sample_indices = [71, 77, 85, 89] + \
                            [40, 42, 44, 45, 47, 49] + \
                            [59, 60, 61, 62, 63, 64] + [65, 66, 67, 68, 69, 70] + \
                            [50]
    face_sample_indices = [x - 23 for x in face_sample_indices]

    body_pose = joints[:, :25, :]
    face = joints[:, 67: , :]

    body_pose = body_pose[:, body_sample_indices, :]    
    face = face[:, face_sample_indices, :]

    pose_tuple = (body_pose, joints[:, 25 : 67, :], face)
    pose_cated = np.concatenate(pose_tuple, axis=1) 

    # pad pose
    T, V, C = pose_cated.shape  # Frame, Joint(78), Coord(3)
    # assert T == len(filenames)
    if T < config.get('clip_length', 16):
        diff = config.get('clip_length', 16) - T
        pose_output = np.concatenate(
            (pose_cated, np.zeros((diff, V, C))), axis=0)
    elif T > config.get('clip_length', 16):
        offset = 0
        pose_output = pose_cated[offset: offset +
                                config.get('clip_length', 16), :, :]
    else:
        pose_output = pose_cated


    pose_length = T if T <= config.get('clip_length', 16) else config.get('clip_length', 16)

    visual_prefix, visual_length = pose_output, pose_length

    visual_prefix[:, :, :2] = dataset.normalize_joints(visual_prefix[:, :, :2])
    # reorder [T V C] -> [C T V]
    visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
    visual_prefix = torch.from_numpy(visual_prefix)
    visual_prefix = visual_prefix.type(torch.FloatTensor)


    gen_text_list = []

    with torch.no_grad():
        B = visual_prefix.shape[0]

        visual_length = torch.tensor([pose_length], dtype=torch.int64)

        visual_prefix = visual_prefix.unsqueeze(0)

        visual_prefix, visual_length = visual_prefix.to(
            device), visual_length.to(device)

        predicted_token_ids = model( # prediction line 
            x=visual_prefix,
            x_length=visual_length,
            phase='test',
        )


        hyposis_list = tokenizer.batch_decode(
            predicted_token_ids, skip_special_tokens=True)
        gen_text_list.extend(hyposis_list)

        # print("All generated hypotheses:")
        print(gen_text_list)

    # End timer and print evaluation speed
    eval_end_time = time.time()
    print(f"Evaluation time (seconds): {eval_end_time - eval_start_time:.4f}")

    return gen_text_list

if __name__ == "__main__":
    input_data = {
        "keypoints_name": "_2FBDaOPYig_1-3-rgb_front",
    }
    run_inference(input_data)
