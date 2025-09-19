import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from typing import List, Optional, Tuple, Union, Dict, Any

def init_independent_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    mesh_dim_names: Optional[Tuple[str, ...]] = None,
    mesh_id: Optional[str] = None  # New parameter to make each mesh unique
) -> DeviceMesh:
    """
    Drop-in replacement for dist.init_device_mesh that creates truly independent
    process groups for each mesh instance.
    
    Args:
        device_type: Device type (e.g., "cuda")
        mesh_shape: Shape of the mesh (e.g., (pp_size, ep_size, fsdp_size))
        mesh_dim_names: Names for mesh dimensions (e.g., ("pp", "ep", "fsdp"))
        mesh_id: Unique identifier for this mesh (e.g., f"mb_{i}")
    
    Returns:
        DeviceMesh with independent process groups
    """
    if mesh_id is None:
        # Generate a unique mesh ID based on current timestamp and rank
        import time
        mesh_id = f"mesh_{int(time.time() * 1000000)}_{dist.get_rank()}"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Calculate mesh dimensions
    total_mesh_size = 1
    for dim_size in mesh_shape:
        total_mesh_size *= dim_size
    
    if total_mesh_size != world_size:
        raise ValueError(f"Mesh shape {mesh_shape} doesn't match world size {world_size}")
    
    # Create independent process groups for each dimension
    independent_groups = {}
    
    # Calculate which ranks belong to each dimension group
    mesh_coords = _get_mesh_coordinates(rank, mesh_shape)
    
    for dim_idx, (dim_name, dim_size) in enumerate(zip(mesh_dim_names or range(len(mesh_shape)), mesh_shape)):
        # Calculate ranks that share the same coordinates except for this dimension
        group_ranks = []
        for dim_rank in range(dim_size):
            # Create coordinates with this dimension varying
            coords = list(mesh_coords)
            coords[dim_idx] = dim_rank
            target_rank = _coordinates_to_rank(coords, mesh_shape)
            group_ranks.append(target_rank)
        
        # Create unique process group for this dimension and mesh
        group = dist.new_group(
            ranks=group_ranks,
            backend="nccl",
            group_desc=f"{mesh_id}_{dim_name}_dim{dim_idx}_ranks{group_ranks}"
        )
        independent_groups[dim_name] = group
    
    # Create the mesh using the independent groups
    return _create_custom_device_mesh(
        device_type=device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        independent_groups=independent_groups,
        mesh_id=mesh_id
    )

def _get_mesh_coordinates(rank: int, mesh_shape: Tuple[int, ...]) -> List[int]:
    """Convert flat rank to mesh coordinates"""
    coords = []
    remaining_rank = rank
    
    for dim_size in reversed(mesh_shape):
        coords.append(remaining_rank % dim_size)
        remaining_rank //= dim_size
    
    return list(reversed(coords))

def _coordinates_to_rank(coords: List[int], mesh_shape: Tuple[int, ...]) -> int:
    """Convert mesh coordinates to flat rank"""
    rank = 0
    multiplier = 1
    
    for coord, dim_size in zip(reversed(coords), reversed(mesh_shape)):
        rank += coord * multiplier
        multiplier *= dim_size
    
    return rank

def _create_custom_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    mesh_dim_names: Optional[Tuple[str, ...]], 
    independent_groups: dict,
    mesh_id: str
) -> DeviceMesh:
    """Create a DeviceMesh that uses our independent process groups"""
    
    # Calculate the mesh tensor (rank layout)
    world_size = dist.get_world_size()
    mesh_tensor = torch.arange(world_size).reshape(mesh_shape)
    
    # Create the base DeviceMesh
    mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh_tensor,
        mesh_dim_names=mesh_dim_names
    )
    
    # Override the get_group method to return our independent groups
    original_get_group = mesh.get_group
    
    def get_independent_group(mesh_dim: Union[int, str] = None):
        if mesh_dim is None:
            # Return the default group (entire mesh)
            return dist.group.WORLD
        
        if isinstance(mesh_dim, int):
            if mesh_dim_names and mesh_dim < len(mesh_dim_names):
                dim_name = mesh_dim_names[mesh_dim]
            else:
                dim_name = f"dim_{mesh_dim}"
        else:
            dim_name = mesh_dim
        
        if dim_name in independent_groups:
            return independent_groups[dim_name]
        else:
            # Fallback to original behavior
            return original_get_group(mesh_dim)
    
    # Monkey patch the get_group method
    mesh.get_group = get_independent_group
    
    # Store mesh_id for debugging
    mesh._mesh_id = mesh_id
    
    return mesh

def compare_device_mesh_structures(
    mesh1: DeviceMesh, 
    mesh2: DeviceMesh,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare two DeviceMesh structures to verify they have the same topology
    but potentially different communicators.
    
    Args:
        mesh1: First DeviceMesh to compare
        mesh2: Second DeviceMesh to compare  
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dictionary containing comparison results and details
    """
    comparison_result = {
        "topology_match": True,
        "communicator_match": True,
        "details": {},
        "differences": [],
        "mesh1_info": {},
        "mesh2_info": {}
    }
    
    # Extract mesh information
    mesh1_info = _extract_mesh_info(mesh1, "mesh1")
    mesh2_info = _extract_mesh_info(mesh2, "mesh2")
    
    comparison_result["mesh1_info"] = mesh1_info
    comparison_result["mesh2_info"] = mesh2_info
    
    # Compare basic topology
    topology_comparison = _compare_topology(mesh1_info, mesh2_info)
    comparison_result["details"]["topology"] = topology_comparison
    
    if not topology_comparison["match"]:
        comparison_result["topology_match"] = False
        comparison_result["differences"].extend(topology_comparison["differences"])
    
    # Compare process groups (communicators)
    pg_comparison = _compare_process_groups(mesh1, mesh2, mesh1_info, mesh2_info)
    comparison_result["details"]["process_groups"] = pg_comparison
    
    if pg_comparison["same_communicators"]:
        comparison_result["communicator_match"] = True
        comparison_result["differences"].append("WARNING: Meshes use the same communicators (not independent)")
    else:
        comparison_result["communicator_match"] = False
    
    # Compare rank assignments per dimension
    rank_comparison = _compare_rank_assignments(mesh1, mesh2, mesh1_info, mesh2_info)
    comparison_result["details"]["rank_assignments"] = rank_comparison
    
    if verbose:
        _print_comparison_results(comparison_result)
    
    return comparison_result

def _extract_mesh_info(mesh: DeviceMesh, mesh_name: str) -> Dict[str, Any]:
    """Extract detailed information from a DeviceMesh"""
    info = {
        "name": mesh_name,
        "mesh_id": getattr(mesh, '_mesh_id', 'unknown'),
        "shape": mesh.mesh.shape,
        "size": mesh.size(),
        "ndim": mesh.ndim,
        "device_type": mesh.device_type,
        "mesh_dim_names": mesh.mesh_dim_names,
        "mesh_tensor": mesh.mesh.clone(),
        "rank_coordinates": {},
        "dimension_groups": {}
    }
    
    # Get current rank's coordinates in the mesh
    current_rank = dist.get_rank()
    if current_rank in mesh.mesh.flatten():
        coords = torch.where(mesh.mesh == current_rank)
        if len(coords[0]) > 0:
            info["rank_coordinates"][current_rank] = tuple(coord[0].item() for coord in coords)
    
    # Extract process group information for each dimension
    for dim_idx in range(mesh.ndim):
        try:
            group = mesh.get_group(dim_idx)
            group_ranks = dist.get_process_group_ranks(group) if group != dist.group.WORLD else list(range(dist.get_world_size()))
            
            info["dimension_groups"][dim_idx] = {
                "group_id": id(group),
                "group_ranks": group_ranks,
                "group_size": len(group_ranks),
                "dimension_name": mesh.mesh_dim_names[dim_idx] if mesh.mesh_dim_names else f"dim_{dim_idx}"
            }
        except Exception as e:
            info["dimension_groups"][dim_idx] = {"error": str(e)}
    
    return info

def _compare_topology(info1: Dict, info2: Dict) -> Dict[str, Any]:
    """Compare the basic topology of two meshes"""
    result = {
        "match": True,
        "differences": []
    }
    
    # Compare shapes
    if info1["shape"] != info2["shape"]:
        result["match"] = False
        result["differences"].append(f"Shape mismatch: {info1['shape']} vs {info2['shape']}")
    
    # Compare sizes
    if info1["size"] != info2["size"]:
        result["match"] = False
        result["differences"].append(f"Size mismatch: {info1['size']} vs {info2['size']}")
    
    # Compare dimensions
    if info1["ndim"] != info2["ndim"]:
        result["match"] = False
        result["differences"].append(f"Dimension count mismatch: {info1['ndim']} vs {info2['ndim']}")
    
    # Compare dimension names
    if info1["mesh_dim_names"] != info2["mesh_dim_names"]:
        result["match"] = False
        result["differences"].append(f"Dimension names mismatch: {info1['mesh_dim_names']} vs {info2['mesh_dim_names']}")
    
    # Compare mesh tensors (rank layout)
    if not torch.equal(info1["mesh_tensor"], info2["mesh_tensor"]):
        result["match"] = False
        result["differences"].append("Mesh tensor (rank layout) mismatch")
    
    return result

def _compare_process_groups(mesh1: DeviceMesh, mesh2: DeviceMesh, info1: Dict, info2: Dict) -> Dict[str, Any]:
    """Compare process groups between two meshes"""
    result = {
        "same_communicators": True,
        "dimension_comparisons": {},
        "differences": []
    }
    
    for dim_idx in range(min(info1["ndim"], info2["ndim"])):
        dim_result = {
            "same_group_id": False,
            "same_ranks": False,
            "group1_info": info1["dimension_groups"].get(dim_idx, {}),
            "group2_info": info2["dimension_groups"].get(dim_idx, {})
        }
        
        group1_info = info1["dimension_groups"].get(dim_idx, {})
        group2_info = info2["dimension_groups"].get(dim_idx, {})
        
        # Compare group IDs (different IDs mean different communicators)
        if group1_info.get("group_id") == group2_info.get("group_id"):
            dim_result["same_group_id"] = True
        else:
            result["same_communicators"] = False
            
        # Compare rank assignments
        if group1_info.get("group_ranks") == group2_info.get("group_ranks"):
            dim_result["same_ranks"] = True
        
        result["dimension_comparisons"][dim_idx] = dim_result
    
    return result

def _compare_rank_assignments(mesh1: DeviceMesh, mesh2: DeviceMesh, info1: Dict, info2: Dict) -> Dict[str, Any]:
    """Compare rank assignments across dimensions"""
    result = {
        "rank_topology_match": True,
        "dimension_details": {}
    }
    
    current_rank = dist.get_rank()
    
    for dim_idx in range(min(info1["ndim"], info2["ndim"])):
        dim_name = info1["dimension_groups"].get(dim_idx, {}).get("dimension_name", f"dim_{dim_idx}")
        
        try:
            # Get groups for this dimension
            group1 = mesh1.get_group(dim_idx)
            group2 = mesh2.get_group(dim_idx)
            
            ranks1 = dist.get_process_group_ranks(group1) if group1 != dist.group.WORLD else list(range(dist.get_world_size()))
            ranks2 = dist.get_process_group_ranks(group2) if group2 != dist.group.WORLD else list(range(dist.get_world_size()))
            
            result["dimension_details"][dim_idx] = {
                "dimension_name": dim_name,
                "mesh1_ranks": ranks1,
                "mesh2_ranks": ranks2,
                "ranks_match": ranks1 == ranks2,
                "current_rank_in_both": current_rank in ranks1 and current_rank in ranks2,
                "group1_id": id(group1),
                "group2_id": id(group2),
                "different_communicators": id(group1) != id(group2)
            }
            
            if ranks1 != ranks2:
                result["rank_topology_match"] = False
                
        except Exception as e:
            result["dimension_details"][dim_idx] = {"error": str(e)}
    
    return result

def _print_comparison_results(result: Dict[str, Any]) -> None:
    """Print detailed comparison results"""
    print("\n" + "="*80)
    print("DEVICE MESH COMPARISON RESULTS")
    print("="*80)
    
    mesh1_info = result["mesh1_info"]
    mesh2_info = result["mesh2_info"]
    
    print(f"\nMESH 1 INFO:")
    print(f"  Name: {mesh1_info['name']}")
    print(f"  Mesh ID: {mesh1_info['mesh_id']}")
    print(f"  Shape: {mesh1_info['shape']}")
    print(f"  Dimension Names: {mesh1_info['mesh_dim_names']}")
    print(f"  Mesh Tensor:\n{mesh1_info['mesh_tensor']}")
    
    print(f"\nMESH 2 INFO:")
    print(f"  Name: {mesh2_info['name']}")
    print(f"  Mesh ID: {mesh2_info['mesh_id']}")
    print(f"  Shape: {mesh2_info['shape']}")
    print(f"  Dimension Names: {mesh2_info['mesh_dim_names']}")
    print(f"  Mesh Tensor:\n{mesh2_info['mesh_tensor']}")
    
    print(f"\nTOPOLOGY MATCH: {'✅ YES' if result['topology_match'] else '❌ NO'}")
    print(f"INDEPENDENT COMMUNICATORS: {'✅ YES' if not result['communicator_match'] else '❌ NO (same communicators)'}")
    
    if result["differences"]:
        print(f"\nDIFFERENCES:")
        for diff in result["differences"]:
            print(f"  - {diff}")
    
    print(f"\nDIMENSION-BY-DIMENSION ANALYSIS:")
    rank_details = result["details"]["rank_assignments"]["dimension_details"]
    
    for dim_idx, details in rank_details.items():
        if "error" in details:
            print(f"  Dimension {dim_idx}: ERROR - {details['error']}")
            continue
            
        dim_name = details["dimension_name"]
        print(f"\n  Dimension {dim_idx} ({dim_name}):")
        print(f"    Mesh1 ranks: {details['mesh1_ranks']}")
        print(f"    Mesh2 ranks: {details['mesh2_ranks']}")
        print(f"    Ranks match: {'✅' if details['ranks_match'] else '❌'}")
        print(f"    Different communicators: {'✅' if details['different_communicators'] else '❌'}")
        print(f"    Group IDs: {details['group1_id']} vs {details['group2_id']}")
    
    print("\n" + "="*80)

# Utility function to verify independent meshes
def verify_independent_meshes(meshes: List[DeviceMesh], verbose: bool = True) -> bool:
    """
    Verify that a list of meshes have the same topology but independent communicators
    
    Args:
        meshes: List of DeviceMesh objects to verify
        verbose: Whether to print detailed results
        
    Returns:
        True if all meshes have same topology but independent communicators
    """
    if len(meshes) < 2:
        if verbose:
            print("Need at least 2 meshes to compare")
        return True
    
    all_independent = True
    reference_mesh = meshes[0]
    
    if verbose:
        print(f"\nVerifying {len(meshes)} meshes for independence...")
    
    for i, mesh in enumerate(meshes[1:], 1):
        result = compare_device_mesh_structures(reference_mesh, mesh, verbose=False)
        
        topology_ok = result["topology_match"]
        communicators_independent = not result["communicator_match"]
        
        if verbose:
            print(f"Mesh 0 vs Mesh {i}: Topology={'✅' if topology_ok else '❌'}, Independent={'✅' if communicators_independent else '❌'}")
        
        if not topology_ok or not communicators_independent:
            all_independent = False
            if verbose:
                print(f"  Issues: {result['differences']}")
    
    if verbose:
        print(f"\nOverall result: {'✅ All meshes are independent' if all_independent else '❌ Some meshes share communicators or have different topologies'}")
    
    return all_independent
