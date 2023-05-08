# Convert output of fusibile to DTU evaluation format.
# By: Jiayu Yang
# Date: 2020-03-30

import os
from os import listdir

fusibile_out_folder="/media/yons/10T1/wanglina/casmvsnetfse/CasMVSNet/outputs/"
dtu_eval_folder="/media/yons/10T1/wanglina/casmvsnetfse/CasMVSNet/eth3d_eval/"

if not os.path.isdir(dtu_eval_folder):
    os.mkdir(dtu_eval_folder)

# Read test list
testlist = "/media/yons/10T1/wanglina/casmvsnetfse/CasMVSNet/lists/dtu/test.txt"
with open(testlist) as f:
    scans = f.readlines()
    scans = [line.rstrip() for line in scans]

for scan in scans:
    # Move ply to dtu eval folder and rename
    scan_folder = os.path.join(fusibile_out_folder,scan,"points_mvsnet")
    print(scan_folder)
    consis_folders = [f for f in listdir(scan_folder) if f.startswith('consistencyCheck-')]
    consis_folders.sort()
    consis_folder = consis_folders[-1]
    source_ply = os.path.join(scan_folder,consis_folder,'final3d_model.ply')
    scan_idx = scan
    name=scan+".ply"
    target_ply = os.path.join(dtu_eval_folder,name)

    cmd = 'cp '+source_ply+' '+target_ply

    print(cmd)
    os.system(cmd)
