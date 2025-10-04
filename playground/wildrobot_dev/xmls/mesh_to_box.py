#!/usr/bin/env python3
"""
mesh_to_box.py

Replace mesh-type geoms in an MJCF XML with primitive box geoms computed from the mesh.

Usage:
    python mesh_to_box.py --xml /path/to/model.xml --geom <geom_xpath_or_name> --mode [aabb|obb] [--padding 0.02] [--mesh-root /path/to/meshes]

Examples:
    # Replace geom with name "left_foot" using OBB (PCA)
    python mesh_to_box.py --xml wild_robot_dev.xml --geom left_foot --mode obb

    # Replace multiple geoms via XPath (all geom elements under a body)
    python mesh_to_box.py --xml model.xml --geom "worldbody/body/body/geom" --mode aabb
"""

import argparse
import shutil
import sys
from pathlib import Path
import math
import numpy as np
import trimesh
from lxml import etree

# -------------------------
# Utilities
# -------------------------
def rotation_matrix_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    # Numerically robust conversion
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)

def mesh_aabb(mesh, padding=0.01):
    """Compute AABB center and half-sizes from a trimesh.Trimesh mesh."""
    v = mesh.vertices
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center = 0.5 * (vmin + vmax)
    half_sizes = 0.5 * (vmax - vmin)
    half_sizes = half_sizes * (1.0 + padding)
    return center, half_sizes

def mesh_obb_pca(mesh, padding=0.01):
    """Compute OBB via PCA. Returns center (world), half_sizes (along local axes), quat (w,x,y,z)."""
    v = mesh.vertices
    centroid = v.mean(axis=0)
    vc = v - centroid
    # covariance and eigenvectors
    C = np.cov(vc.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    # sort by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # rotate points into PCA frame
    v_pca = (eigvecs.T @ vc.T).T
    pmin = v_pca.min(axis=0)
    pmax = v_pca.max(axis=0)
    half_sizes = 0.5 * (pmax - pmin)
    center_pca = 0.5 * (pmax + pmin)
    center_world = eigvecs @ center_pca + centroid
    half_sizes = half_sizes * (1.0 + padding)
    quat = rotation_matrix_to_quat(eigvecs)
    return center_world, half_sizes, quat

# -------------------------
# XML helpers
# -------------------------
NS = None  # MuJoCo MJCF typically has no namespace; adjust if needed

def find_mesh_file_from_asset(root, mesh_name):
    """
    Look for <asset><mesh name="mesh_name" file="..."/> and return the file attribute.
    Return None if not found.

    """
    return mesh_name + ".stl"
    # assets = root.findall('.//mesh')  # search anywhere for mesh tags
    # for m in assets:
    #     if m.get('name') == mesh_name:
    #         f = m.get('file') or m.get('src')  # some MJCF may use different attr
    #         if f:
    #             return f
    return None

def find_target_geoms(root, geom_path_or_name):
    """
    If geom_path_or_name looks like an XPath (contains '/', '[', or ends with 'geom'),
    treat as XPath and select matching elements. Otherwise treat as geom name and select
    geoms with matching @name.
    """
    # Simple heuristic
    if ('/' in geom_path_or_name) or ('[' in geom_path_or_name) or geom_path_or_name.endswith('geom'):
        # treat as XPath - support simple relative XPaths from root
        try:
            geoms = root.xpath(geom_path_or_name)
            # filter to element type geom
            geoms = [g for g in geoms if etree.iselement(g) and g.tag.endswith('geom')]
            return geoms
        except Exception:
            # fall back to name search
            pass
    # treat as geom name
    geoms = root.findall('.//geom')
    selected = [g for g in geoms if g.get('name') == geom_path_or_name or g.get('mesh') == geom_path_or_name]
    return selected

def replace_geom_with_box(elem, pos, half_sizes, quat=None, keep_attrs=None):
    """
    Replace an existing geom element (elem) with a box geom.
    - pos: 3-array
    - half_sizes: 3-array (hx,hy,hz)
    - quat: optional quaternion (w,x,y,z). If None, emit axis-aligned box (no quat).
    - keep_attrs: list of attribute names from original geom to copy (like contype/conaffinity/group/class/name)
    """
    parent = elem.getparent()
    # create new element
    new = etree.Element('geom')
    new.set('type', 'box')
    pos_str = f"{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}"
    size_str = f"{half_sizes[0]:.6g} {half_sizes[1]:.6g} {half_sizes[2]:.6g}"
    new.set('pos', pos_str)
    new.set('size', size_str)
    if quat is not None:
        quat_str = f"{quat[0]:.6g} {quat[1]:.6g} {quat[2]:.6g} {quat[3]:.6g}"
        new.set('quat', quat_str)
    # preserve some attributes like contype/conaffinity/class/name/group/rgba
    if keep_attrs:
        for a in keep_attrs:
            v = elem.get(a)
            if v is not None:
                new.set(a, v)
    # Try to preserve ordering: replace in parent
    idx = parent.index(elem)
    parent.remove(elem)
    parent.insert(idx, new)
    return new

# -------------------------
# Main processing
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Replace mesh geoms in MJCF with AABB or OBB boxes.")
    p.add_argument('--xml', required=True, help='Path to MJCF XML file')
    p.add_argument('--geom', required=True, help='Geom selector: either geom name or an XPath (e.g. "//body/body/geom")')
    p.add_argument('--mode', choices=['aabb', 'obb'], default='aabb', help='Compute AABB or OBB (PCA)')
    p.add_argument('--padding', type=float, default=0.02, help='Padding fraction to inflate half-sizes (default 0.02 = 2%%)')
    p.add_argument('--mesh-root', default=None, help='Optional root directory to resolve mesh asset file paths')
    p.add_argument('--keep', default='contype,conaffinity,class,name,group,rgba', help='Comma-separated geom attributes to preserve')
    args = p.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        print("XML file not found:", xml_path, file=sys.stderr)
        sys.exit(2)

    # Backup original
    backup_path = xml_path.with_suffix(xml_path.suffix + '.bak')
    shutil.copy2(xml_path, backup_path)
    print(f"Backup saved to: {backup_path}")

    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    # find target geoms
    geoms = find_target_geoms(root, args.geom)
    if not geoms:
        print("No geoms found for selector:", args.geom, file=sys.stderr)
        sys.exit(3)
    print(f"Found {len(geoms)} geom(s) matching selector.")

    keep_attrs = [k.strip() for k in args.keep.split(',') if k.strip()]

    # For each geom, try to find mesh reference and process
    updated = 0
    for g in geoms:
        mesh_name = g.get('mesh')
        print("Processing geom:", g.get('name'), "mesh:", mesh_name)
        mesh_file = None
        if mesh_name:
            # try find in asset by name
            mesh_file = find_mesh_file_from_asset(root, mesh_name)
            if mesh_file is None:
                # mesh attribute might be a direct file path
                mesh_file = mesh_name
        else:
            # maybe the geom references an asset via 'name' attribute or something else - fallback to name
            print(f"Warning: geom has no 'mesh' attribute: {etree.tostring(g, pretty_print=True).decode('utf8')}", file=sys.stderr)

        if mesh_file is None:
            print("Could not determine mesh file for geom. Skipping this geom.", file=sys.stderr)
            continue

        # resolve relative path
        if args.mesh_root:
            mesh_path = Path(args.mesh_root) / mesh_file
        else:
            mesh_path = (xml_path.parent / mesh_file).resolve()

        if not mesh_path.exists():
            print(f"Mesh file not found for geom: {mesh_path}. Skipping.", file=sys.stderr)
            continue

        print(f"Processing geom -> mesh file: {mesh_path}")

        # load mesh via trimesh
        try:
            mesh = trimesh.load(str(mesh_path), force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                # scene -> merge
                mesh = trimesh.util.concatenate(mesh.dump())
        except Exception as e:
            print("Error loading mesh:", e, file=sys.stderr)
            continue

        # If the mesh vertices are in a different frame because the MJCF mesh asset had a pos/quat/euler,
        # you should apply that transform here. Currently we assume mesh file is already in correct local frame.

        if args.mode == 'aabb':
            center, half_sizes = mesh_aabb(mesh, padding=args.padding)
            quat = None
        else:
            center, half_sizes, quat = mesh_obb_pca(mesh, padding=args.padding)

        # Replace geom element
        new_geom = replace_geom_with_box(g, pos=center, half_sizes=half_sizes, quat=quat, keep_attrs=keep_attrs)
        print("Replaced geom with box (pos,size,quat):", new_geom.get('pos'), new_geom.get('size'), new_geom.get('quat'))
        updated += 1

    if updated:
        # write out (pretty)
        tree.write(str(xml_path), pretty_print=True, xml_declaration=True, encoding='utf-8')
        print(f"Updated {updated} geom(s) in {xml_path}")
    else:
        print("No geoms were updated.")

if __name__ == '__main__':
    main()
