import torch
import time
import GPUtil
import threading
from tqdm import tqdm

# Configuration
GPU_INDICES = [1, 2, 3]  # List of GPUs to occupy (e.g., [5,6,7])
CHUNK_SIZE = 100 * 1024**2  # 10 MiB chunks
SAFETY_MARGIN = 110 * 1024**2  # 50 MiB safety buffer
COMPUTE_INTERVAL = 0.1  # Seconds between computations


def validate_gpus():
    """Check requested GPUs exist and are available"""
    available_gpus = [gpu.id for gpu in GPUtil.getGPUs()]
    valid_gpus = [gpu for gpu in GPU_INDICES if gpu in available_gpus]

    if not valid_gpus:
        raise ValueError("No valid GPUs available from specified indices!")
    return valid_gpus


def occupy_and_compute(gpu_id, stop_event):
    """Occupy GPU memory and perform light computations"""
    device = torch.device(f"cuda:{gpu_id}")
    allocated_memory = []

    try:
        # Light computation setup
        a = torch.randn(256, 256, device=device)  # Small matrix
        b = torch.randn(256, 256, device=device)

        while not stop_event.is_set():
            # Occupy remaining memory
            gpu = GPUtil.getGPUs()[gpu_id]
            available_memory = gpu.memoryFree * 1024**2 - SAFETY_MARGIN

            if available_memory >= CHUNK_SIZE:
                # Allocate memory in chunks
                chunk = torch.empty(CHUNK_SIZE, dtype=torch.uint8, device=device)
                allocated_memory.append(chunk)
                tqdm.write(
                    f"GPU {gpu_id}: Allocated {CHUNK_SIZE//1024**2} MiB | Free: {available_memory//1024**2} MiB"
                )

            # Perform light computation
            a = torch.mm(a, b)  # Matrix multiplication
            a = a * 0.9 + torch.randn_like(a) * 0.1  # Noise injection

            # Regulate computation frequency
            time.sleep(COMPUTE_INTERVAL)

    except Exception as e:
        tqdm.write(f"GPU {gpu_id} error: {str(e)}")
    finally:
        # Cleanup
        del allocated_memory
        torch.cuda.empty_cache()
        tqdm.write(f"GPU {gpu_id}: Memory released and computation stopped")


def monitor_usage(stop_event):
    """Periodically display GPU status"""
    while not stop_event.is_set():
        try:
            gpus = GPUtil.getGPUs()
            status = []
            for gpu in gpus:
                if gpu.id in GPU_INDICES:
                    status.append(
                        f"GPU {gpu.id}: {gpu.load*100:.1f}% load | {gpu.memoryUsed}MB used"
                    )
            tqdm.write("\t".join(status))
            time.sleep(2)
        except:
            break


def main():
    valid_gpus = validate_gpus()
    print(f"Occupying and computing on GPUs: {valid_gpus}")

    stop_event = threading.Event()
    threads = []

    # Start worker threads for each GPU
    for gpu_id in valid_gpus:
        thread = threading.Thread(
            target=occupy_and_compute, args=(gpu_id, stop_event), daemon=True
        )
        thread.start()
        threads.append(thread)

    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_usage, args=(stop_event,), daemon=True
    )
    monitor_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping workers...")
        stop_event.set()
        for thread in threads:
            thread.join()
        monitor_thread.join()
        print("Cleanup complete")


if __name__ == "__main__":
    main()