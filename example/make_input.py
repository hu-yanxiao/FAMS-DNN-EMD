import json
import os
from ase.io import read
import shutil
import ptagnn

def modify_json_file(file_path, key, new_value):
    """
    修改 JSON 文件中指定键的值

    参数:
        file_path (str): JSON 文件路径
        key (str): 要修改的键
        new_value: 新的值
    """
    # 读取 JSON 文件
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 修改指定键的值
    if key in data:
        data[key] = new_value
    else:
        print(f"警告: 键 '{key}' 不存在于 JSON 文件中")

    # 写回文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
def add_to_lsf_file(file_path, line_to_add):
    with open(file_path, 'a') as f:  # 'a' 表示追加模式
        f.write(line_to_add)  # 确保写入后换行

if __name__ == "__main__":

    name = "normal"
    model_type = "gnn"
    fullname = name+"_"+model_type
    dirpath = "./"
    jsonfile_static = "./input_static.json"
    jsonfile = jsonfile_static.replace("static",name)
    if model_type == "gnn":
        jsonfile = jsonfile.replace(".json","_gnn.json")
    elif model_type == "linear":
        jsonfile = jsonfile.replace(".json", "_linear.json")
    shutil.copyfile(jsonfile_static,jsonfile)
    train_file = r"/work/phy-qianwx/model/testhyx_1/train_data0.xyz"
    test_file = r"/work/phy-qianwx/model/testhyx_1/test_data0.xyz"
    valid_file = r"/work/phy-qianwx/model/testhyx_1/valid_data0.xyz"
    modify_json_file(jsonfile, "train_file", train_file)
    modify_json_file(jsonfile, "test_file", test_file)
    modify_json_file(jsonfile, "valid_file", valid_file)
    modify_json_file(jsonfile, "model_type", model_type)
    # 根据任务属性的处理
    modify_json_file(jsonfile, "name", fullname)
    if name == "target":
        modify_json_file(jsonfile, "loss", name)
        energy_key = input("Please input the key of property: ")
        modify_json_file(jsonfile, "energy_key", energy_key)
        modify_json_file(jsonfile, "teacher_modelpath", "null")
    elif name == "distill":
        teacher_model = input("Please input teacher model fullpath: ")
        modify_json_file(jsonfile, "teacher_modelpath", teacher_model)
        modify_json_file(jsonfile, "energy_key", "energy")
    elif name == "normal":
        print("Task of energy/force/stress/virvals fit")
        modify_json_file(jsonfile, "energy_key", "energy")
        modify_json_file(jsonfile, "teacher_modelpath", "null")
    else:
        print("name should be target/distill/normal")

    #输出文件路径处理
    log_dir = os.path.join(dirpath,"logs")
    modify_json_file(jsonfile,"log_dir", log_dir)
    model_dir = os.path.join(dirpath, "model")
    modify_json_file(jsonfile, "model_dir", model_dir)
    checkpoints_dir = os.path.join(dirpath, "checkpoints")
    modify_json_file(jsonfile, "checkpoints_dir", checkpoints_dir)
    results_dir = os.path.join(dirpath, "results")
    modify_json_file(jsonfile, "results_dir", results_dir)
    downloads_dir = os.path.join(dirpath, "downloads")
    modify_json_file(jsonfile, "downloads_dir", downloads_dir)
    # 根据文件判定loss
    if name != "target":
        atoms_first = read(train_file,index=1)
        print(atoms_first.arrays.keys())
        print(atoms_first.info.keys())
        if not "forces" in atoms_first.arrays.keys():
            print("no forces for not target task, if only need fit energy use name as target and energy_key energy")
        else:
            if "stress" in atoms_first.info.keys():
                modify_json_file(jsonfile, "loss", "universal")
                modify_json_file(jsonfile, "compute_stress", True)
            elif "virials" in atoms_first.info.keys():
                modify_json_file(jsonfile, "loss", "virials")
                modify_json_file(jsonfile, "compute_stress", True)
            elif "virial" in atoms_first.info.keys():
                modify_json_file(jsonfile, "loss", "virials")
                modify_json_file(jsonfile, "compute_stress", True)
            else:
                print("no stress and virials")
                if "energy" in atoms_first.info.keys():
                    modify_json_file(jsonfile, "loss", "weighted")
                else:
                    print("have forces but no energy,stress and virials")
                    modify_json_file(jsonfile, "compute_stress", False)
    if model_type == "gnn":
        num_interactions = 2
        modify_json_file(jsonfile, "num_interactions", num_interactions)
        gate = "tanh"
        modify_json_file(jsonfile, "gate", gate)
    elif model_type == "linear":
        num_interactions = 1
        modify_json_file(jsonfile, "num_interactions", num_interactions)
        gate = 'None'
        modify_json_file(jsonfile, "gate", gate)
    else:
        print("model_type should be gnn or linear")
    
    if name == "normal":
        runpy = ptagnn.__file__.replace("__init__.py","cli/my_run_train.py")
    elif name == "target":
        runpy = ptagnn.__file__.replace("__init__.py","cli/my_run_train_target.py")
    elif name == "distill":
        runpy = ptagnn.__file__.replace("__init__.py","cli/my_run_train_distill.py")

    runpystring = f"python {runpy} --input {jsonfile}"
    sourcelsf = 'MLP_empty.lsf'
    marklsf = name+"_"+model_type+".lsf"
    dstlsf = sourcelsf.replace("empty.lsf",marklsf)
    shutil.copyfile('MLP_empty.lsf',dstlsf)
    add_to_lsf_file(dstlsf, runpystring)



