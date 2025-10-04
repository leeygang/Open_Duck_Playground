import os, trimesh


meshdir = "assets"
for f in os.listdir(meshdir):
    if f.endswith(".stl") or f.endswith(".obj"):
        m = trimesh.load(os.path.join(meshdir,f), force='mesh')
        print(f, "bytes:", os.path.getsize(os.path.join(meshdir,f)), "tri:", len(m.faces))
