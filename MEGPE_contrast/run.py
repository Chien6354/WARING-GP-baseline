import subprocess

start_num = 1
end_num = 25

for i in range(start_num, end_num + 1):
    log_file = f"log/main_{i}.log"
    pkl_file = f"log/latest_population_{i}.pkl"

    command = ["python", "main.py", "--log_file", log_file, "--pkl_file", pkl_file]

    print(f"Running: {' '.join(command)}")
    subprocess.run(command)
