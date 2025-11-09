import os
import subprocess
import multiprocessing as mp
import time
import logging

# ========== 日志配置 ==========
log_dir = "/root/autodl-tmp/sr/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"batch_train_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 基础配置 ==========
base_train_script = "/root/autodl-tmp/sr/SeeSR/train_dreambooth_lora_textual_inversion.py"
base_dataset_root = "/root/autodl-tmp/sr/datasets/train/dataset/"
output_dir_root = "/root/autodl-tmp/sr/models/output_1109_batch/dreambooth"
os.makedirs(output_dir_root, exist_ok=True)

# 类别列表
categories = sorted(os.listdir(base_dataset_root))
gpus = ["0"]  # 你只有一张 GPU，所以多进程共用同一 GPU
group_size = 3  # ✅ 每次并行 3 个类别

# ========== 辅助函数 ==========
def run_training(cls_name, gpu_id):
    """运行单个类别训练"""
    # 每个类别单独日志
    cls_log = os.path.join(log_dir, f"train_{cls_name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    cls_logger = logging.getLogger(cls_name)
    cls_handler = logging.FileHandler(cls_log, encoding="utf-8")
    cls_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    cls_logger.addHandler(cls_handler)
    cls_logger.setLevel(logging.INFO)

    cls_logger.info(f"🟢 启动类别：{cls_name} (GPU {gpu_id})")

    instance_data_dir = os.path.join(base_dataset_root, cls_name, "train")
    ti_initializer_token = cls_name
    instance_prompt = f"a photo of sks {cls_name}"
    validation_prompt = f"a boy with a sks {cls_name}"

    pattern = "lora-dreambooth-ti-model"
    model_name = f"{pattern}-{cls_name}"
    output_dir = os.path.join(output_dir_root, model_name)

    cmd = [
        "python", base_train_script,
        "--ti_initializer_token", ti_initializer_token,
        "--instance_data_dir", instance_data_dir,
        "--instance_prompt", instance_prompt,
        "--validation_prompt", validation_prompt,
        "--output_dir", output_dir,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cls_logger.info(f"🚀 GPU {gpu_id} 运行命令: {' '.join(cmd)}")
    start_time = time.time()
    subprocess.run(cmd, env=env)
    end_time = time.time()

    duration = (end_time - start_time) / 60
    cls_logger.info(f"✅ 类别 {cls_name} 训练完成 (GPU {gpu_id})，耗时 {duration:.2f} 分钟")
    cls_logger.removeHandler(cls_handler)
    cls_handler.close()


# ========== 主流程 ==========
def main():
    logger.info("============== 启动批量训练（每次并行 3 类） ==============")
    logger.info(f"总类别数: {len(categories)} | 日志文件: {log_file}\n")

    for i in range(0, len(categories), group_size):
        group = categories[i:i + group_size]
        logger.info(f"🚀 启动批次：{group}")

        processes = []
        start_batch_time = time.time()

        for cls in group:
            p = mp.Process(target=run_training, args=(cls, gpus[0]))
            p.start()
            processes.append(p)

        # 等这一批的 3 个进程都训练完
        for p in processes:
            p.join()

        end_batch_time = time.time()
        batch_dur = (end_batch_time - start_batch_time) / 60
        logger.info(f"✅ 批次 {group} 完成，耗时 {batch_dur:.2f} 分钟，等待 10 秒进入下一个批次...\n")
        time.sleep(10)

    logger.info("🎉 全部类别训练完成！")


if __name__ == "__main__":
    main()
