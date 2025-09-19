import json
import os
from tqdm import tqdm
import pandas as pd

current_dir = os.getcwd()
target_dir = os.path.join(current_dir, 'cves')

cve_data = pd.DataFrame(columns=['cve', 'owner', 'repo', 'desc', 'commit_id'])

for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith('.json'):
            print(f"current file: {file}")
            file_path = os.path.join(root, file)
            # dir_name = os.path.basename(os.path.dirname(root))
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            cve_id = data['cveMetadata']['cveId']

            # 有些cve的描述或者ref可能为空
            if 'descriptions' in data['containers']['cna'] and 'references' in data['containers']['cna']:
                desc = data['containers']['cna']['descriptions'][0]['value']
                references = data['containers']['cna']['references']
            else:
                continue

            # if 'descriptions' in data:
            #     desc = data['containers']['cna']['descriptions'][0]['value']
            # else:
            #     desc = 'NAN'
            # references = data['containers']['cna']['references']

            ref_list = []  # 防止有多个链接
            for ref in references:
                if ref['url'].startswith("https://github.com") and len(ref['url'].split("/")) == 7:
                    owner = ref['url'].split("/")[3]
                    repo = ref['url'].split("/")[4]
                    commit_id = ref['url'].split("/")[6]
                    if len(commit_id) == 40:
                        ref_list.append({"owner": owner, "repo": repo, "commit_id": commit_id})
                    else:
                        continue

            if not ref_list:
                continue
            else:
                for ref_num in range(len(ref_list)):
                    cve_data.loc[len(cve_data)] = [cve_id, ref_list[ref_num]['owner'], ref_list[ref_num]['repo'], desc,
                                                   ref_list[ref_num]['commit_id']]

cve_data.to_csv(os.path.join(current_dir, 'cve_data.csv'), index=False)
