# #!/usr/bin/env python3
# import argparse
# import csv
# import os
# import sys
# import time
# import subprocess
# import psutil
#
# MB = 1024 * 1024
#
# def to_mb(x):
#     return round(x / MB, 2) if x is not None else ""
#
# def collect_procs_list(pid, include_children=False):
#     try:
#         p = psutil.Process(pid)
#     except psutil.NoSuchProcess:
#         return [], {"rss": 0, "vms": 0, "private": 0, "threads": 0}
#
#     procs = [p]
#     if include_children:
#         try:
#             procs += p.children(recursive=True)
#         except Exception:
#             pass
#
#     total_rss = 0
#     total_vms = 0
#     total_private = 0
#     entries = []
#     for pr in procs:
#         try:
#             mi = pr.memory_info()
#             rss = getattr(mi, "rss", 0) or 0
#             vms = getattr(mi, "vms", 0) or 0
#             try:
#                 mfi = pr.memory_full_info()
#                 private = getattr(mfi, "private", 0) or 0
#             except Exception:
#                 private = 0
#             entries.append({
#                 "pid": pr.pid,
#                 "rss": rss,
#                 "vms": vms,
#                 "private": private,
#                 "name": pr.name() if hasattr(pr, "name") else ""
#             })
#             total_rss += rss
#             total_vms += vms
#             total_private += private
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#
#     totals = {
#         "rss": total_rss,
#         "vms": total_vms,
#         "private": total_private,
#         "threads": len(entries)
#     }
#     return entries, totals
#
# def make_header(max_procs):
#     base = [
#         "iso_time", "elapsed_s", "entry_type",
#         "pid", "rss_MiB", "vms_MiB", "private_MiB", "processes"
#     ]
#     proc_cols = []
#     for i in range(1, max_procs + 1):
#         proc_cols += [f"proc_{i}_pid", f"proc_{i}_rss_MiB", f"proc_{i}_vms_MiB"]
#     return base + proc_cols
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--save", "-s", required=True, help="CSV 输出路径")
#     parser.add_argument("--interval", "-i", type=float, default=1.0, help="采样间隔（秒），默认 1.0")
#     parser.add_argument("--include-children", action="store_true", help="包含子进程（parent + children）")
#     parser.add_argument("--max-procs", type=int, default=16)
#     parser.add_argument("cmd", nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#
#     cmd_list = args.cmd or []
#
#     # 移除开头的 '--' 分隔符（如果存在）
#     if len(cmd_list) > 0 and cmd_list[0] == "--":
#         cmd_list = cmd_list[1:]
#
#     if not cmd_list:
#         print("没有提供要运行的命令。请在 --save 后面提供要执行的命令。", file=sys.stderr)
#         sys.exit(2)
#
#     # 如果第一个元素是一个 .py 文件路径，使用当前 Python 解释器启动它
#     first = cmd_list[0]
#     if isinstance(first, str) and first.lower().endswith(".py"):
#         cmd = [sys.executable] + cmd_list
#     else:
#         # 如果第一个元素看起来像 python 可执行（例如 C:\...\python.exe）或其它命令，直接使用用户提供的列表
#         cmd = cmd_list
#
#     print("启动命令：", cmd)
#     try:
#         proc = subprocess.Popen(cmd, shell=False)
#     except FileNotFoundError:
#         print("无法启动命令，命令：{}".format(cmd), file=sys.stderr)
#         sys.exit(1)
#     except Exception as e:
#         print("启动命令异常：", e, "命令：", cmd, file=sys.stderr)
#         sys.exit(1)
#
#     print("已启动进程 PID:", proc.pid)
#     start_ts = time.time()
#     next_tick = start_ts
#     peak_total_rss = 0
#     warned_overflow = False  # 仅告警一次
#
#     header = make_header(args.max_procs)
#     os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
#     with open(args.save, "w", newline="", encoding="utf-8-sig") as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#
#         try:
#             while True:
#                 now = time.time()
#                 iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
#                 elapsed = now - start_ts
#
#                 procs_entries, totals = collect_procs_list(proc.pid, include_children=args.include_children)
#
#                 row = [
#                     iso,
#                     f"{elapsed:.3f}",
#                     "TOTAL",
#                     proc.pid,
#                     to_mb(totals["rss"]),
#                     to_mb(totals["vms"]),
#                     (to_mb(totals["private"]) if totals["private"] > 0 else ""),
#                     totals["threads"]
#                 ]
#
#                 if len(procs_entries) > args.max_procs and not warned_overflow:
#                     print(f"[WARN] 进程数 {len(procs_entries)} 超过 --max-procs {args.max_procs}，将只记录前 {args.max_procs} 个进程。")
#                     warned_overflow = True
#
#                 for i in range(args.max_procs):
#                     if i < len(procs_entries):
#                         e = procs_entries[i]
#                         row += [e["pid"], to_mb(e["rss"]), to_mb(e["vms"])]
#                     else:
#                         row += ["", "", ""]
#
#                 writer.writerow(row)
#                 f.flush()
#
#                 if totals["rss"] and totals["rss"] > peak_total_rss:
#                     peak_total_rss = totals["rss"]
#
#                 if proc.poll() is not None:
#                     break
#
#                 next_tick += args.interval
#                 time.sleep(max(0.0, next_tick - time.time()))
#
#         except KeyboardInterrupt:
#             try:
#                 proc.terminate()
#             except Exception:
#                 pass
#
#     peak_total_rss_mb = round(peak_total_rss / MB, 2)
#     print("监控结束。峰值 TOTAL RSS（MiB）:", peak_total_rss_mb)
#     print("CSV 已保存到:", os.path.abspath(args.save))
#
# if __name__ == "__main__":
#     main()
import argparse
import csv
import os
import sys
import time
import subprocess
import psutil

MB = 1024 * 1024

def build_command(cmd_list):
    if cmd_list and isinstance(cmd_list[0], str) and cmd_list[0].lower().endswith(".py"):
        return [sys.executable] + cmd_list
    return cmd_list

def to_mb(x):
    return round(x / MB, 2) if x is not None else ""

def collect_procs_list(pid, include_children=False):

    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return [], {"rss": 0, "vms": 0, "private": 0, "threads": 0}

    procs = [p]
    if include_children:
        try:
            procs += p.children(recursive=True)
        except Exception:
            pass

    total_rss = 0
    total_vms = 0
    total_private = 0
    total_threads = 0

    entries = []
    for pr in procs:
        try:
            mi = pr.memory_info()
            rss = getattr(mi, "rss", 0) or 0
            vms = getattr(mi, "vms", 0) or 0
            threads = pr.num_threads() if hasattr(pr, "num_threads") else 0
            private = 0
            try:
                mfi = pr.memory_full_info()
                private = getattr(mfi, "private", 0) or 0
            except Exception:
                private = 0
            entries.append({
                "pid": pr.pid,
                "rss": rss,
                "vms": vms,
                "threads": threads,
                "private": private,
                "name": (pr.name() if hasattr(pr, "name") else "")
            })
            total_rss += rss
            total_vms += vms
            total_private += private
            total_threads += threads
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    totals = {
        "rss": total_rss,
        "vms": total_vms,
        "private": total_private,
        "threads": len(entries)
    }
    return entries, totals

def make_header(max_procs):
    base = [
        "iso_time", "elapsed_s", "entry_type",
        "pid", "rss_MiB", "vms_MiB", "private_MiB", "processes"
    ]
    proc_cols = []
    for i in range(1, max_procs + 1):
        proc_cols += [f"proc_{i}_pid", f"proc_{i}_rss_MiB", f"proc_{i}_vms_MiB"]
    return base + proc_cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", "-s", required=True, help="CSV 输出路径")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="采样间隔（秒），默认 1.0")
    parser.add_argument("--include-children", action="store_true", help="包含子进程（parent + children）")
    parser.add_argument("--max-procs", type=int, default=16)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd_list = args.cmd or []

    # 移除开头的 '--' 分隔符（如果存在）
    if len(cmd_list) > 0 and cmd_list[0] == "--":
        cmd_list = cmd_list[1:]

    if not cmd_list:
        print("请在 --save 之后提供要运行的命令，例如：python monitor_proc_expanded.py --save 1.csv --include-children 被监控代码.py --被监控代码参数", file=sys.stderr)
        sys.exit(2)
    cmd = cmd_list
    # cmd = build_command(args.cmd)
    try:
        proc = subprocess.Popen(cmd, shell=False)
    except FileNotFoundError:
        print("无法启动命令，命令：{}".format(cmd), file=sys.stderr)
        sys.exit(1)

    print("已启动进程 PID:", proc.pid)
    start_ts = time.time()
    next_tick = start_ts
    peak_total_rss = 0
    warned_overflow = False  # 仅告警一次

    header = make_header(args.max_procs)
    with open(args.save, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        try:
            while True:
                now = time.time()
                iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                elapsed = now - start_ts

                procs_entries, totals = collect_procs_list(proc.pid, include_children=args.include_children)

                row = [
                    iso,
                    f"{elapsed:.3f}",
                    "TOTAL",
                    proc.pid,
                    to_mb(totals["rss"]),
                    to_mb(totals["vms"]),
                    (to_mb(totals["private"]) if totals["private"] > 0 else ""),
                    totals["threads"]
                ]

                if len(procs_entries) > args.max_procs and not warned_overflow:
                    print(f"[WARN] 进程数 {len(procs_entries)} 超过 --max-procs {args.max_procs}，将只记录前 {args.max_procs} 个进程。")
                    warned_overflow = True

                for i in range(args.max_procs):
                    if i < len(procs_entries):
                        e = procs_entries[i]
                        row += [e["pid"], to_mb(e["rss"]), to_mb(e["vms"])]
                    else:
                        row += ["", "", ""]

                writer.writerow(row)
                f.flush()

                if totals["rss"] and totals["rss"] > peak_total_rss:
                    peak_total_rss = totals["rss"]


                if proc.poll() is not None:
                    break

                next_tick += args.interval
                time.sleep(max(0.0, next_tick - time.time()))

        except KeyboardInterrupt:
            try:
                proc.terminate()
            except Exception:
                pass

    peak_total_rss_mb = round(peak_total_rss / MB, 2)
    print("监控结束。峰值 TOTAL RSS（MiB）:", peak_total_rss_mb)
    print("CSV 已保存到:", os.path.abspath(args.save))


if __name__ == "__main__":
    main()
