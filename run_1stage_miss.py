import os
import sys

sys.path.append('./utils')
from utils.config_utils import ID_PATH, HEIGHT, WIDTH

if __name__ == "__main__":
    log_dir = sys.argv[1]

    # lines_list = [
    #     ("Drawer", 275),
    #     ("Drawer", 288),
    #     ("Drawer", 290),
    #     ("Drawer", 292),
    #     ("Drawer", 304),
    #     ("Drawer", 309),
    #     ("Drawer", 311),
    #     ("Drawer", 312),
    #     ("Drawer", 313),
    # ]
    
    lines_list = [
        ("Box", 58)
    ]

    total_to_render = len(lines_list)
    cnt = 0
    
    for line in lines_list:
        print(f'Still to run: {total_to_render-cnt}\n')

        category = line[0]
        id = line[1]
        
        # TODO: test
        # category = 'StorageFurniture'
        # id = 41083
        # category = 'CoffeeMachine'
        # id = 103048
        # category = 'Phone'
        # id = 103941
        # category = 'Camera'
        # id = 101352
        # category = 'Door'
        # id = 9288
        # category = 'Safe'
        # id = 102381
        # category = 'Oven'
        # id = 101930
        # category = "Drawer"
        # id = 312
        
        os.system(f'python -u render_annotate.py \
                    --model_id {id} --category {category} --height {HEIGHT} --width {WIDTH} \
                    2>&1 | tee -a {log_dir}')

        print(f'Run Over: {category} : {id}\n')
        cnt += 1
        
        # break
        
    print("All Over!!!")
