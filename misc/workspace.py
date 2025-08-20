import shutil
import datetime
import os
import json

def create_workspace(args):
    """
    创建工作区目录并返回路径
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"{args.model_id}_{timestamp}"
    workspace_dir = os.path.join("./output", workspace_name)
    
    os.makedirs(workspace_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(workspace_dir, "checkpoints")
    logs_dir = os.path.join(workspace_dir, "logs")
    configs_dir = os.path.join(workspace_dir, "configs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    
    return workspace_dir, checkpoints_dir, logs_dir, configs_dir

def save_configs_and_args(args, configs_dir, configs_path):
    """
    保存配置文件和参数到工作区
    """
    # 保存args
    args_dict = vars(args)
    args_path = os.path.join(configs_dir, "args.json")
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)
    
    # 复制configs文件
    if os.path.exists(configs_path):
        config_filename = "configs.json"
        dest_config_path = os.path.join(configs_dir, config_filename)
        shutil.copy2(configs_path, dest_config_path)
        print(f"Configs copied to: {dest_config_path}")
    
    print(f"Args saved to: {args_path}")