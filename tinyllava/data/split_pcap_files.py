import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

def split_single_pcap(args):
    pcap_file, input_dir, output_dir, splitter = args
    pcap_file_path = os.path.join(input_dir, pcap_file)
    base_name = pcap_file[:-5]
    output_folder = os.path.join(output_dir, base_name)
    os.makedirs(output_folder, exist_ok=True)

    cmd = [splitter, "-i", pcap_file_path, "-o", output_folder, "-p", f"{base_name}-", "-f", "five_tuple"]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        # 失败则删除整个文件夹
        shutil.rmtree(output_folder, ignore_errors=True)

    return pcap_file, result.returncode


def split_pcap_files(input_dir, output_dir, splitter, workers=1, verbose=False):
    pcap_files = [f for f in os.listdir(input_dir) if f.endswith(".pcap")]

    if not pcap_files:
        print(f"[!] 未在 {input_dir} 中找到 .pcap 文件")
        return

    print(f"[*] 共发现 {len(pcap_files)} 个 PCAP 文件，使用 {workers} 个并发线程\n")

    success, failed = 0, []

    tasks = [(f, input_dir, output_dir, splitter) for f in pcap_files]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(split_single_pcap, t): t[0] for t in tasks}

        with tqdm(total=len(pcap_files), desc="切分进度", unit="file", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                pcap_file = futures[future]
                try:
                    name, returncode = future.result()
                    if returncode == 0:
                        success += 1
                        if verbose:
                            tqdm.write(f"  [✓] {name}")
                    else:
                        failed.append(name)
                        tqdm.write(f"  [✗] {name} (返回码: {returncode})")
                except Exception as e:
                    failed.append(pcap_file)
                    tqdm.write(f"  [✗] {pcap_file} 发生异常: {e}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"成功": success, "失败": len(failed)})

    print(f"\n[完成] 成功: {success} | 失败: {len(failed)}")
    if failed:
        print("[失败列表]")
        for f in failed:
            print(f"  - {f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量切分 PCAP 文件（按五元组流）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir",  required=True,  help="输入 PCAP 文件夹路径")
    parser.add_argument("-o", "--output-dir", required=True,  help="输出文件夹根路径")
    parser.add_argument(
        "-s", "--splitter",
        default="/root/autodl-tmp/third_party/ShieldGPT-master/pcap_tool/splitter",
        help="splitter 工具路径",
    )
    parser.add_argument("-w", "--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示每个文件的处理结果")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_pcap_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splitter=args.splitter,
        workers=args.workers,
        verbose=args.verbose,
    )