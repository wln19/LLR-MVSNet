import open3d as o3d
import trimesh
import numpy as np
def catmesh(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))

    trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                            vertex_normals=np.asarray(mesh.vertex_normals))
    trimesh.convex.is_convex(trimesh)
    o3d.visualization.draw_geomatrics(trimesh)
    print("success")
if __name__=="__main__":
    path='/media/yons/10T1/wanglina/casmvsnetfse/CasMVSNet/outputs/Family.ply'
    catmesh(path)
