import csv

# 要转换的文件路径
file_paths = ["/home/linxw/yzu-rza/code-xyf/result/code_msg_ commitbart/test_0.output", "/home/linxw/yzu-rza/code-xyf/result/code_msg_ commitbart/test_0.gold"]

for path in file_paths:
    csv_path = path + ".csv"  # 输出文件名
    with open(path, "r", encoding="utf-8", errors="replace") as infile, \
         open(csv_path, "w", newline="", encoding="utf-8") as outfile:
        
        # 按制表符读取
        tsv_reader = csv.reader(infile, delimiter="\t")
        # 按逗号写入
        csv_writer = csv.writer(outfile)
        
        for row in tsv_reader:
            csv_writer.writerow(row)
    
    print(f"已生成: {csv_path}")
