import os
import re

def parse_links(filepath):
    links = set()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            src = parts[0]
            for i in range(1, len(parts), 2):
                dst = parts[i]
                try:
                    ett = float(parts[i+1])
                except (IndexError, ValueError):
                    continue
                # 2000.0等大数视为不可用链路
                if ett < 1000:
                    links.add((src, dst))
    return links

dir_path = 'data/ucsb/1143927049-1143953729'
files = [f for f in os.listdir(dir_path) if f.startswith('neighbortable-')]
# 提取时间戳并排序
files = sorted(files, key=lambda x: int(re.findall(r'neighbortable-(\d+)', x)[0]))

prev_links = None
change_timestamps = []
all_timestamps = []

for fname in files:
    ts = int(re.findall(r'neighbortable-(\d+)', fname)[0])
    all_timestamps.append(ts)
    links = parse_links(os.path.join(dir_path, fname))
    if prev_links is not None and links != prev_links:
        change_timestamps.append(ts)
    prev_links = links

print(f"Total timestamps: {len(files)}")
print(f"Topology changed {len(change_timestamps)} times.")
print("Timestamps with topology change:")
for ts in change_timestamps:
    print(ts) 