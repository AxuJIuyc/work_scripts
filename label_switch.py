import os
import json


def label_switch(path2files):
    files = os.listdir(path2files)
    for file in files:
        if file.endswith("json"):
            with open(path2files + file, 'r') as f:
                jsn = json.load(f)
            for label in jsn["shapes"]:
                if label["label"] == 'obj':
                    label["label"] = 'frutilad'
            with open(f"{path2files}frutilad_box/{file}", 'w') as f:        
                json.dump(jsn, f)

def label_del(path2files):
    files = os.listdir(path2files)
    for file in files:
        if file.endswith("json"):
            with open(path2files + file, 'r') as f:
                jsn = json.load(f)
            del_list = []
            for label, i in zip(jsn["shapes"], range(len(jsn["shapes"]))):
                if label["label"] == 'box':
                    del_list.append(i)
            for index in del_list[::-1]:
                jsn["shapes"].pop(index)
            with open(f"datasets/frut_without_box/{file}", 'w') as f:        
                json.dump(jsn, f)

# path2files = "datasets/frutolad/frutilad_box/"
path2files = "datasets/20.03.23/"
label_del(path2files)
print("Complete!")

